# DL4S-Tensorboard

A summary writer for [DL4S](https://github.com/palle-k/DL4S) that writes Tensorboard log files.

### Installation

Add DL4S-Tensorboard as a package dependency in `Package.swift`.

    .package(url: "https://github.com/palle-k/DL4S-Tensorboard.git", .branch("master")),

Then add `DL4STensorboard` as a target dependency:

    .target(name: "YourAwesomeTarget", dependencies: ["DL4STensorboard"]),

### Usage

Currently, only scalars are supported.

```swift
import DL4STensorboard
import Foundation

let logdir = URL(fileURLWithPath: "./logs")

let writer = try TensorboardWriter(logDirectory: logdir, runName: "Classifier")

try writer.add(scalar: 101, withTag: "model/accuracy", atStep: 1337)
```

