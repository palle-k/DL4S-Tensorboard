# DL4S-Tensorboard

A summary writer for [DL4S](https://github.com/palle-k/DL4S) that writes Tensorboard log files.

### Installation

Add DL4S-Tensorboard as a package dependency in `Package.swift`.

    .package(url: "https://github.com/palle-k/DL4S-Tensorboard.git", .branch("master")),

Then add `DL4STensorboard` as a target dependency:

    .target(name: "YourAwesomeTarget", dependencies: ["DL4STensorboard"]),

### Usage

DL4S-Tensorboard supports Scalars, Images, Tensors, Text, Embeddings and Histograms.

#### Writing scalars

```swift
import DL4STensorboard
import Foundation

let logdir = URL(fileURLWithPath: "./logs")

let writer = try TensorboardWriter(logDirectory: logdir, runName: "Classifier")

try writer.write(scalar: 101, withTag: "model/accuracy", atStep: 1337)
```

#### Advanced Usage

```
// writing an image
try writer.write(image: imageTensor, withTag: "generator/output", atStep: 42)

// writing text
try writer.write(text: "Lorem ipsum dolor sit amet", withTag: "lm/sample", atStep: 314)

// writing embeddings
let embeddingLayer = DL4S.Embedding<Float, CPU>(inputFeatures: 42, outputSize: 128)
try writer.write(embedding: embeddingLayer.embeddingMatrix, withLabels: vocab, atStep: 1337)

// writing a histogram
let histogram = Histogram(values: valueDistribution, buckets: 10)
try writer.write(histogram: histogram, withTag: "data/histogram", atStep: 4242)
```
