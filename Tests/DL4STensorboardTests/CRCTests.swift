//
//  CRCTests.swift
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

import Foundation
import Testing
@testable import DL4STensorboard

@Suite
struct CRCTests {
    /// The CRC-32C check value from the RFC 3720 test vectors.
    @Test
    func matchesCRC32CTestVectors() {
        #expect(crc32(Data()) == 0)
        #expect(crc32(Data(repeating: 0, count: 32)) == 0x8A91_36AA)
        #expect(crc32(Data(repeating: 0xFF, count: 32)) == 0x62A8_AB43)
        #expect(crc32(Data("123456789".utf8)) == 0xE306_9283)
    }

    @Test
    func masksTheCRC() {
        let data = Data("123456789".utf8)
        let crc = crc32(data)
        #expect(masked_crc32c(data) == ((crc >> 15) | (crc << 17)) &+ 0xA282_EAD8)
    }
}
