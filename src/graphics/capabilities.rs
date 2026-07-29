// Adapter capability negotiation: what optional GPU features are available,
// what the engine asked for, and what it actually got.
//
// Ray tracing is the only optional feature so far, and it is the reason this
// module exists as its own unit: the decision "trace or rasterise" has to be
// made once, at device creation, and then reported honestly to every consumer
// (renderer, HUD, console). Deriving it ad hoc from `device.features()` at each
// call site would make a silent fallback indistinguishable from a supported run.

/// Environment variable that forces the ray-tracing path off even on hardware
/// that supports it.
///
/// The no-ray-query code path must stay exercised, and the only honest way to
/// exercise it on a machine that *does* have ray query is to be able to switch
/// the capability off from outside the binary — editing a constant and
/// rebuilding tests a different binary than the one that ships.
pub const NO_RAYTRACING_ENV: &str = "ASTRAGLYPH_NO_RAYTRACING";

/// Why the ray-tracing path is or is not active.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RayTracingStatus {
    /// Ray query is available and enabled on the device.
    Enabled,
    /// The adapter does not expose `EXPERIMENTAL_RAY_QUERY`.
    UnsupportedAdapter,
    /// The adapter supports it, but `NO_RAYTRACING_ENV` asked for the fallback.
    DisabledByEnv,
    /// The adapter advertised it, but `request_device` refused it anyway.
    DeviceRefused,
}

impl RayTracingStatus {
    /// Whether the traced path may be used.
    pub fn is_enabled(self) -> bool {
        matches!(self, Self::Enabled)
    }

    /// One-line explanation, for the startup log and the console.
    pub fn describe(self) -> &'static str {
        match self {
            Self::Enabled => "hardware ray query enabled",
            Self::UnsupportedAdapter => "adapter has no EXPERIMENTAL_RAY_QUERY, using CPU fallback",
            Self::DisabledByEnv => "disabled by ASTRAGLYPH_NO_RAYTRACING, using CPU fallback",
            Self::DeviceRefused => "device refused EXPERIMENTAL_RAY_QUERY, using CPU fallback",
        }
    }

    /// Short tag for the HUD, where there is one cell per character.
    pub fn tag(self) -> &'static str {
        match self {
            Self::Enabled => "RTX",
            Self::UnsupportedAdapter => "NO-HW",
            Self::DisabledByEnv => "OFF-ENV",
            Self::DeviceRefused => "REFUSED",
        }
    }
}

/// Decide whether to *request* ray query, given what the adapter advertises and
/// whether the environment vetoed it.
///
/// Split out from device creation so the decision is testable without a GPU:
/// the fallback branch is the one that must never regress, and it is exactly the
/// branch that cannot be reached on the development machine.
pub fn should_request_ray_query(adapter_supports: bool, vetoed: bool) -> bool {
    adapter_supports && !vetoed
}

/// Classify the outcome after `request_device` returned.
///
/// `granted` is what the *device* reports, not what was asked for — a device
/// may legally hand back fewer features than requested, and that must read as
/// `DeviceRefused` rather than as success.
pub fn classify(adapter_supports: bool, vetoed: bool, granted: bool) -> RayTracingStatus {
    if !adapter_supports {
        RayTracingStatus::UnsupportedAdapter
    } else if vetoed {
        RayTracingStatus::DisabledByEnv
    } else if granted {
        RayTracingStatus::Enabled
    } else {
        RayTracingStatus::DeviceRefused
    }
}

/// TLAS instance slots the engine wants, matching the scene pass's object budget.
///
/// This has to be *requested*: every acceleration-structure limit in
/// `wgpu::Limits::default()` is zero, so a device created with the defaults
/// rejects `create_tlas` outright even on hardware with full ray-tracing
/// support. The failure message ("Limit `max_tlas_instance_count` is 0") points
/// at the limit rather than at the feature, which is easy to misread as "this
/// GPU cannot do it".
pub const REQUESTED_TLAS_INSTANCES: u32 = 1024;

/// Triangles allowed in a single mesh's BLAS. Generous: the cost of a high limit
/// is nothing until a BLAS actually gets that big.
pub const REQUESTED_BLAS_PRIMITIVES: u32 = 1 << 21;

/// Geometry groups per BLAS. One — each mesh is a single triangle group.
pub const REQUESTED_BLAS_GEOMETRIES: u32 = 1;

/// Acceleration structures bound per shader stage. One — the scene TLAS.
pub const REQUESTED_ACCELERATION_STRUCTURES: u32 = 1;

/// What to ask for: what the engine needs, never more than the adapter allows.
///
/// Requesting above the adapter's maximum makes `request_device` fail, which
/// would turn a machine with *smaller* ray-tracing limits into a machine with
/// *no* graphics at all.
pub fn requested_limit(needed: u32, adapter_max: u32) -> u32 {
    needed.min(adapter_max)
}

/// Whether the environment asked for the fallback path.
pub fn ray_tracing_vetoed() -> bool {
    match std::env::var(NO_RAYTRACING_ENV) {
        Ok(value) => {
            let v = value.trim();
            !(v.is_empty() || v == "0" || v.eq_ignore_ascii_case("false"))
        }
        Err(_) => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unsupported_adapter_is_never_requested() {
        assert!(!should_request_ray_query(false, false));
        assert!(!should_request_ray_query(false, true));
    }

    #[test]
    fn supported_adapter_is_requested_unless_vetoed() {
        assert!(should_request_ray_query(true, false));
        assert!(!should_request_ray_query(true, true));
    }

    /// The four outcomes must stay distinguishable: "this GPU cannot" and
    /// "I turned it off" and "the driver lied" are different bug reports.
    #[test]
    fn classify_separates_every_reason() {
        assert_eq!(classify(false, false, false), RayTracingStatus::UnsupportedAdapter);
        assert_eq!(classify(false, true, false), RayTracingStatus::UnsupportedAdapter);
        assert_eq!(classify(true, true, false), RayTracingStatus::DisabledByEnv);
        assert_eq!(classify(true, false, true), RayTracingStatus::Enabled);
        assert_eq!(classify(true, false, false), RayTracingStatus::DeviceRefused);
    }

    /// A device that quietly grants nothing must not be reported as enabled —
    /// this is the case that would otherwise send ray queries at a device
    /// without the feature and trip validation at draw time.
    #[test]
    fn granted_false_is_not_enabled() {
        assert!(!classify(true, false, false).is_enabled());
        assert!(classify(true, false, true).is_enabled());
    }

    /// A requested limit must never exceed what the adapter reports, or the
    /// device request fails and the application has no GPU at all.
    #[test]
    fn requested_limits_never_exceed_the_adapter() {
        assert_eq!(requested_limit(1024, 1 << 24), 1024);
        assert_eq!(requested_limit(1024, 256), 256);
        assert_eq!(requested_limit(1024, 0), 0);
    }

    /// The zero defaults are the trap this module exists to document: a nonzero
    /// request is mandatory even on a fully capable adapter.
    #[test]
    fn requested_acceleration_limits_are_all_nonzero() {
        for value in [
            REQUESTED_TLAS_INSTANCES,
            REQUESTED_BLAS_PRIMITIVES,
            REQUESTED_BLAS_GEOMETRIES,
            REQUESTED_ACCELERATION_STRUCTURES,
        ] {
            assert!(value > 0, "an acceleration limit of 0 rejects every TLAS");
        }
    }

    #[test]
    fn only_enabled_status_permits_tracing() {
        for status in [
            RayTracingStatus::UnsupportedAdapter,
            RayTracingStatus::DisabledByEnv,
            RayTracingStatus::DeviceRefused,
        ] {
            assert!(!status.is_enabled(), "{status:?} must not permit tracing");
            assert!(!status.describe().is_empty());
            assert!(!status.tag().is_empty());
        }
    }
}
