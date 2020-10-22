//
//  TSVWriter.swift
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

import Foundation


class TSVWriter {
    let target: URL
    let handle: FileHandle
    
    init(target: URL) throws {
        self.target = target
        if !FileManager.default.fileExists(atPath: target.path) {
            FileManager.default.createFile(atPath: target.path, contents: nil, attributes: nil)
        }
        self.handle = try FileHandle(forWritingTo: target)
    }
    
    func writeHeader(columns: [String]) throws {
        try self.handle.seek(toOffset: 0)
        handle.write(columns.joined(separator: "\t") + "\n")
    }
    
    func writeRow(entries: [Any]) throws {
        if #available(OSX 10.15.4, iOS 13.4, watchOS 6.2, tvOS 13.4, *) {
            try self.handle.seekToEnd()
        } else {
            self.handle.seekToEndOfFile()
        }
        handle.write(entries.map{"\($0)"}.joined(separator: "\t") + "\n")
    }
    
    func close() throws {
        try self.handle.synchronize()
        try self.handle.close()
    }
}
