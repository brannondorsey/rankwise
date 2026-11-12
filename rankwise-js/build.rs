fn main() {
    // Node-API addons link against symbols the host process provides (macOS needs
    // `-undefined dynamic_lookup`); napi-build sets that up.
    #[cfg(feature = "node")]
    napi_build::setup();
}
