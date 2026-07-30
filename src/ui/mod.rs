// UI module: the interactive layers drawn on top of the 3D scene.
//
// Everything here is pure logic plus drawing into an `ascii::Overlay`. There is
// deliberately no winit or GPU dependency: the app layer translates real key
// presses into the abstract actions these modules accept, which keeps UI
// behaviour unit-testable and the bindings replaceable.

pub mod console;
pub mod editor;
pub mod menu;

#[allow(unused_imports)]
pub use console::{Console, ConsoleAction, ConsoleLine, LineKind};
#[allow(unused_imports)]
pub use menu::{Menu, MenuAction, MenuEvent, MenuItem};
