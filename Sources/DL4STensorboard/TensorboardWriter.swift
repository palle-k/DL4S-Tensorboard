//
//  TensorboardWriter.swift
//
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
import SwiftProtobuf


public class TensorboardWriter {
    private let handle: FileHandle
    
    public init(logDirectory: URL, runName: String?) throws {
        let logfileName = "events.out.tfevents.\(Int(Date().timeIntervalSince1970)).\(Host.current().name ?? "localhost")"
        let runDirectory: URL
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
    
    private func writeInitEvent() throws {
        var event = Tensorflow_Event()
        event.wallTime = Date().timeIntervalSince1970
        try write(event: event)
    }
    
    public func write(scalar: Float, withTag tag: String, atStep step: Int) throws {
        var value = Tensorflow_Summary.Value()
        value.simpleValue = scalar
        value.tag = cleanTag(tag)
        
        var summary = Tensorflow_Summary()
        summary.value = [value]
        
        var event = Tensorflow_Event()
        event.step = Int64(step)
        event.wallTime = Date().timeIntervalSince1970
        event.what = .summary(summary)
        
        try write(event: event)
    }
    
    public func write<Device>(image: Tensor<Float, Device>) {
        
    }
}
