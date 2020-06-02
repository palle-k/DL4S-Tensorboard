//
//  TensorboardWriter.swift
//
//  Created by Palle Klewitz on 02.06.20.
//  Copyright (c) 2020 Palle Klewitz
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
import SwiftGD
import SwiftProtobuf


/// A summary writer that writes TensorBoard compatible log files
public class TensorboardWriter {
    private let runDirectory: URL
    private let handle: FileHandle
    
    /// Creates a summary writer that writes TensorBoard compatible log files
    /// - Parameters:
    ///   - logDirectory: Directory to write log files into
    ///   - runName: Name of the current run. If specified, a subdirectory with the run name will be used.
    /// - Throws: An error if the events file could not be created
    public init(logDirectory: URL, runName: String?) throws {
        let logfileName = "events.out.tfevents.\(Int(Date().timeIntervalSince1970)).\(Host.current().name ?? "localhost")"
        if let runName = runName {
            runDirectory = logDirectory.appendingPathComponent(runName, isDirectory: true)
        } else {
            runDirectory = logDirectory
        }
        let fileURL = runDirectory.appendingPathComponent(logfileName)
        
        if !FileManager.default.fileExists(atPath: runDirectory.path) {
            try FileManager.default.createDirectory(atPath: runDirectory.path, withIntermediateDirectories: true, attributes: nil)
        }
        if !FileManager.default.fileExists(atPath: fileURL.path) {
            FileManager.default.createFile(atPath: fileURL.path, contents: nil, attributes: nil)
        }
        
        handle = try FileHandle(forUpdating: fileURL)
        
        try writeInitEvent()
    }
    
    private func write(event: Tensorflow_Event) throws {
        try handle.seekToEnd()
        let eventData = try event.serializedData()
        let size = UInt64(eventData.count)
        let header = Data(integer: size)
        
        handle.write(header)
        handle.write(Data(integer: masked_crc32c(header)))
        
        handle.write(eventData)
        handle.write(Data(integer: masked_crc32c(eventData)))
        try handle.synchronize()
    }
    
    private func write(value: Tensorflow_Summary.Value, atStep step: Int) throws {
        var summary = Tensorflow_Summary()
        summary.value = [value]
        
        var event = Tensorflow_Event()
        event.step = Int64(step)
        event.wallTime = Date().timeIntervalSince1970
        event.what = .summary(summary)
        
        try write(event: event)
    }
    
    private func writeInitEvent() throws {
        var event = Tensorflow_Event()
        event.wallTime = Date().timeIntervalSince1970
        try write(event: event)
    }
    
    /// Writes a single scalar to tensorboard.
    /// - Parameters:
    ///   - scalar: Scalar to write
    ///   - tag: Tag for the scalar
    ///   - step: Current epoch/step/training iteration
    /// - Throws: An error if the writer was unable to write to disk
    public func write(scalar: Float, withTag tag: String, atStep step: Int) throws {
        var value = Tensorflow_Summary.Value()
        value.simpleValue = scalar
        value.tag = cleanTag(tag)
        
        try write(value: value, atStep: step)
    }
    
    /// Writes an image to tensorboard
    /// - Parameters:
    ///   - image: Tensor containing the image data, either [channels, height, width] or [height, width], with the number of supported channels being either 1 (grayscale), 3 (rgb) or 4 (rgba).
    ///   - tag: Tag for the image
    ///   - step: Current epoch/step/training iteration
    /// - Throws: An error if the writer was unable to write to disk
    public func write<Element, Device>(image: Tensor<Element, Device>, withTag tag: String, atStep step: Int) throws {
        guard let gdImage = Image(image) else {
            print("Could not create image from tensor.")
            return
        }
        let pngData = try gdImage.export(as: .png)
        
        var imageValue = Tensorflow_Summary.Image()
        imageValue.colorspace = 4
        imageValue.width = Int32(gdImage.size.width)
        imageValue.height = Int32(gdImage.size.height)
        imageValue.encodedImageString = pngData
        
        var value = Tensorflow_Summary.Value()
        value.image = imageValue
        value.tag = cleanTag(tag)
        
        try write(value: value, atStep: step)
    }
    
    /// Writes a tensor to tensorboard
    /// - Parameters:
    ///   - tensor: Arbitrary tensor
    ///   - tag: Tag for the tensor
    ///   - step: Current epoch/step/training iteration
    /// - Throws: An error if the writer was unable to write to disk
    public func write<Element: TensorFlowProtoScalar, Device>(tensor: Tensor<Element, Device>, withTag tag: String, atStep step: Int) throws {
        let tensorShape = Tensorflow_TensorShapeProto.with { tensorShape in
            tensorShape.dim = tensor.shape.map { v in
                Tensorflow_TensorShapeProto.Dim.with { dim in
                    dim.size = Int64(v)
                }
            }
        }
        let tensorProto = Tensorflow_TensorProto.with { tensorProto in
            tensorProto.dtype = Element.dtype.dtype
            tensorProto.tensorShape = tensorShape
            var wrapper = TensorProtoWrapper(tensor: tensorProto)
            Element.populate(tensorProto: &wrapper, with: tensor)
            tensorProto = wrapper.tensor
        }
    
        var value = Tensorflow_Summary.Value()
        value.tensor = tensorProto
        value.tag = cleanTag(tag)
    
        try write(value: value, atStep: step)
    }
    
    /// Writes text to tensorboard
    /// - Parameters:
    ///   - text: Text data that will be written to TensorBoard
    ///   - tag: Tag for the text
    ///   - step: Current epoch/step/training iteration
    /// - Throws: An error if the writer was unable to write to disk
    public func write(text: String, withTag tag: String, atStep step: Int) throws {
        let tensorShape = Tensorflow_TensorShapeProto.with {
            $0.dim = [Tensorflow_TensorShapeProto.Dim.with {$0.size = 1}]
        }
        let tensor = Tensorflow_TensorProto.with {
            $0.tensorShape = tensorShape
            $0.dtype = .dtString
            $0.stringVal = [text.data(using: .utf8) ?? Data()]
        }
        let pluginData = Tensorflow_SummaryMetadata.PluginData.with {
            $0.pluginName = "text"
        }
        let meta = Tensorflow_SummaryMetadata.with {
            $0.pluginData = [pluginData]
        }
        
        var value = Tensorflow_Summary.Value()
        value.tensor = tensor
        value.tag = cleanTag(tag)
        value.metadata = meta
        
        try write(value: value, atStep: step)
    }
    
    /// Writes a histogram to tensorboard
    /// - Parameters:
    ///   - histogram: Histogram data
    ///   - tag: Tag for the histogram
    ///   - step: Current epoch/step/training iteration
    /// - Throws: An error if the writer was unable to write to disk
    public func write(histogram: Histogram, withTag tag: String, atStep step: Int) throws {
        let histogramProto = Tensorflow_HistogramProto.with {
            $0.bucket = histogram.buckets
            $0.bucketLimit = Array(histogram.edges.dropFirst())
            $0.min = histogram.min
            $0.max = histogram.max
            $0.sum = histogram.sum
            $0.sumSquares = zip(histogram.buckets, histogram.buckets).map(*).reduce(0, +)
            $0.num = histogram.sum
        }
        
        let value = Tensorflow_Summary.Value.with {
            $0.histo = histogramProto
            $0.tag = cleanTag(tag)
        }
        
        try write(value: value, atStep: step)
    }
    
    /// Writes an embedding matrix to tensorboard
    /// - Parameters:
    ///   - embedding: Tensor with shape [items, embedDim]
    ///   - labels: Labels corresponding to rows in the embedding matrix
    ///   - tag: Tag for the image
    ///   - step: Current epoch/step/training iteration
    /// - Throws: An error if the writer was unable to write to disk
    public func write<Element, Device>(embedding: Tensor<Element, Device>, withLabels labels: [String], atStep step: Int) throws {
        precondition(embedding.dim == 2, "Embedding must be 2-dimensional tensor")
        precondition(embedding.shape[0] == labels.count, "Number of labels must be equal to number of rows in embedding tensor.")
        
        let paddedGlobalStep = String(format: "%05d", step)
        
        let embeddingDir = runDirectory.appendingPathComponent(paddedGlobalStep)
        try FileManager.default.createDirectory(at: embeddingDir, withIntermediateDirectories: true, attributes: nil)
        
        try labels.joined(separator: "\n")
            .write(to: embeddingDir.appendingPathComponent("metadata.tsv"), atomically: true, encoding: .utf8)
        
        let tensorsWriter = try TSVWriter(target: embeddingDir.appendingPathComponent("tensors.tsv"))
        for rowIndex in 0 ..< embedding.shape[0] {
            let row = embedding[rowIndex]
            try tensorsWriter.writeRow(entries: row.elements)
        }
        try tensorsWriter.close()
        
        let projectorConfigURL = runDirectory.appendingPathComponent("projector_config.pbtxt")
        if !FileManager.default.fileExists(atPath: projectorConfigURL.path) {
            FileManager.default.createFile(atPath: projectorConfigURL.path, contents: nil, attributes: nil)
        }
        let projectorConfigFile = try FileHandle(forUpdating: projectorConfigURL)
        projectorConfigFile.seekToEndOfFile()
        projectorConfigFile.write("""
        embeddings {
        tensor_name: "default:\(paddedGlobalStep)"
        tensor_path: "\(paddedGlobalStep)/tensors.tsv"
        metadata_path: "\(paddedGlobalStep)/metadata.tsv"
        }
        
        """) // do not remove last newline from the string.
        try projectorConfigFile.synchronize()
        try projectorConfigFile.close()
    }
}
