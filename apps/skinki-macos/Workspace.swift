import ProjectDescription

// Workspace groups the app project together with the local Swift packages so
// they show up side by side in Xcode after `tuist generate`.
let workspace = Workspace(
    name: "Skinki",
    projects: [
        ".",
    ]
)
