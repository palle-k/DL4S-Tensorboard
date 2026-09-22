//
//  TensorboardWriterTests.swift
//
//  Created by Palle Klewitz on 21.09.26.
//  Copyright (c) 2026 Palle Klewitz
//
//  Permission is hereby granted, free of charge, to any person obtaining a copy
//  of this software and associated documentation files (the "Software"), to deal
//  in the Software without restriction, including without limitation the rights
//  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
//  copies of the Software, and to permit persons to whom the Software is
//  furnished to do so, subject to the following conditions:
//
//  The above copyright notice and this permission notice shall be included in all
//  copies or substantial portions of the Software.
//
//  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
//  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
//  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
//  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
//  SOFTWARE.

import DL4S
import Foundation
import SwiftProtobuf
import Testing
@testable import DL4STensorboard

/// Reads the events back from a TFRecord file and checks the CRC of every record.
private func readEvents(from url: URL) throws -> [Tensorflow_Event] {
    let data = try Data(contentsOf: url)
    var events: [Tensorflow_Event] = []
    var offset = 0
    while offset < data.count {
        let header = data[offset ..< offset + 8]
        let length = Int(header.withUnsafeBytes { $0.loadUnaligned(as: UInt64.self) }.littleEndian)
        let headerCRC = data[offset + 8 ..< offset + 12].withUnsafeBytes { $0.loadUnaligned(as: UInt32.self) }.littleEndian
        #expect(headerCRC == masked_crc32c(Data(header)))
        let payload = Data(data[offset + 12 ..< offset + 12 + length])
        let payloadCRC = data[offset + 12 + length ..< offset + 16 + length].withUnsafeBytes { $0.loadUnaligned(as: UInt32.self) }.littleEndian
        #expect(payloadCRC == masked_crc32c(payload))
        events.append(try Tensorflow_Event(serializedBytes: payload))
        offset += 16 + length
    }
    return events
}

private func eventsFile(in runDirectory: URL) throws -> URL {
    let files = try FileManager.default.contentsOfDirectory(at: runDirectory, includingPropertiesForKeys: nil)
        .filter { $0.lastPathComponent.hasPrefix("events.out.tfevents.") }
    #expect(files.count == 1)
    return try #require(files.first)
}

/// Every test gets its own instance and its own log directory, which `deinit` removes after the test.
@Suite
final class TensorboardWriterTests {
    let logDirectory = FileManager.default.temporaryDirectory.appendingPathComponent("DL4STensorboardTests-\(UUID().uuidString)", isDirectory: true)

    deinit {
        try? FileManager.default.removeItem(at: logDirectory)
    }

    @Test
    func writesScalarsAsRecords() throws {
        let writer = try TensorboardWriter(logDirectory: logDirectory, runName: "scalars")
        for i in 1 ... 10 {
            try writer.write(scalar: log(Float(i)), withTag: "main/loss", atStep: i)
        }

        let events = try readEvents(from: try eventsFile(in: logDirectory.appendingPathComponent("scalars")))
        #expect(events.count == 11)
        #expect(events[0].what == nil)
        for (i, event) in events.dropFirst().enumerated() {
            #expect(event.step == Int64(i + 1))
            let value = try #require(event.summary.value.first)
            #expect(value.tag == "main/loss")
            #expect(value.simpleValue == log(Float(i + 1)))
        }
    }

    @Test
    func writesTextAndHistogram() throws {
        let writer = try TensorboardWriter(logDirectory: logDirectory, runName: nil)
        try writer.write(text: "Lorem ipsum", withTag: "//lm/sample text", atStep: 3)
        let histogram = Histogram(values: [0, 1, 2, 3], buckets: 2)
        try writer.write(histogram: histogram, withTag: "data/histogram", atStep: 4)

        let events = try readEvents(from: try eventsFile(in: logDirectory))
        #expect(events.count == 3)

        let text = try #require(events[1].summary.value.first)
        #expect(text.tag == "lm/sample_text")
        #expect(text.metadata.pluginData.pluginName == "text")
        #expect(text.tensor.stringVal == [Data("Lorem ipsum".utf8)])

        let histo = try #require(events[2].summary.value.first)
        #expect(histo.histo.bucket == [2, 2])
        #expect(histo.histo.num == 4)
        #expect(histo.histo.min == 0)
    }

    @Test
    func writesImagesAndTensors() throws {
        let writer = try TensorboardWriter(logDirectory: logDirectory, runName: "images")
        let grayscale = Tensor<Float, CPU>(repeating: 0.5, shape: 4, 6)
        try writer.write(image: grayscale, withTag: "gray", atStep: 1)
        let rgb = Tensor<Float, CPU>(repeating: 0.25, shape: 3, 5, 7)
        try writer.write(image: rgb, withTag: "rgb", atStep: 2)
        let tensor = Tensor<Float, CPU>([[1, 2, 3], [4, 5, 6]])
        try writer.write(tensor: tensor, withTag: "tensor", atStep: 3)

        let events = try readEvents(from: try eventsFile(in: logDirectory.appendingPathComponent("images")))
        #expect(events.count == 4)
        let gray = try #require(events[1].summary.value.first)
        #expect(gray.image.width == 6)
        #expect(gray.image.height == 4)
        #expect(gray.image.encodedImageString.starts(with: [0x89, 0x50, 0x4E, 0x47]))
        let color = try #require(events[2].summary.value.first)
        #expect(color.image.width == 7)
        #expect(color.image.height == 5)
        let proto = try #require(events[3].summary.value.first)
        #expect(proto.tensor.dtype == .dtFloat)
        #expect(proto.tensor.tensorShape.dim.map(\.size) == [2, 3])
        #expect(proto.tensor.floatVal == [1, 2, 3, 4, 5, 6])
    }

    @Test
    func rejectsUnsupportedImageShapes() throws {
        let writer = try TensorboardWriter(logDirectory: logDirectory, runName: "bad-images")
        let twoChannels = Tensor<Float, CPU>(repeating: 0, shape: 2, 4, 4)
        #expect(throws: TensorboardWriterError.self) {
            try writer.write(image: twoChannels, withTag: "image", atStep: 1)
        }
    }

    @Test
    func writesEmbeddings() throws {
        let writer = try TensorboardWriter(logDirectory: logDirectory, runName: "embeddings")
        let embedding = Tensor<Float, CPU>([[1, 2], [3, 4], [5, 6]])
        try writer.write(embedding: embedding, withLabels: ["a", "b", "c"], atStep: 7)

        let runDirectory = logDirectory.appendingPathComponent("embeddings")
        let metadata = try String(contentsOf: runDirectory.appendingPathComponent("00007/metadata.tsv"), encoding: .utf8)
        #expect(metadata == "a\nb\nc")
        let tensors = try String(contentsOf: runDirectory.appendingPathComponent("00007/tensors.tsv"), encoding: .utf8)
        #expect(tensors == "1.0\t2.0\n3.0\t4.0\n5.0\t6.0\n")
        let config = try String(contentsOf: runDirectory.appendingPathComponent("projector_config.pbtxt"), encoding: .utf8)
        #expect(config.contains("tensor_path: \"00007/tensors.tsv\""))
    }

    @Test
    func sharesOneWriterBetweenTasks() async throws {
        let writer = try TensorboardWriter(logDirectory: logDirectory, runName: "concurrent")
        await withTaskGroup(of: Void.self) { group in
            for task in 0 ..< 8 {
                group.addTask {
                    for step in 0 ..< 25 {
                        try? writer.write(scalar: Float(task), withTag: "task/\(task)", atStep: step)
                    }
                }
            }
        }
        let events = try readEvents(from: try eventsFile(in: logDirectory.appendingPathComponent("concurrent")))
        #expect(events.count == 201)
    }
}
