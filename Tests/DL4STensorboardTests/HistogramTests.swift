//
//  HistogramTests.swift
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
import Testing
@testable import DL4STensorboard

@Suite
struct HistogramTests {
    @Test
    func countsEveryValueOnce() {
        let elements = (0 ..< 100).map { _ in Double(Float.random(in: 0 ... 1)) }
        let histogram = Histogram(values: elements, buckets: 20)
        #expect(histogram.buckets.count == 20)
        #expect(histogram.sum == 100)
        #expect(histogram.edges.count == 21)
        #expect(histogram.edges.first == histogram.min)
        #expect(histogram.edges.last == histogram.max)
    }

    @Test
    func placesTheLargestValueInTheLastBucket() {
        let histogram = Histogram(values: [0, 1, 2, 3, 4], buckets: 5)
        #expect(histogram.buckets == [1, 1, 1, 1, 1])
        #expect(histogram.min == 0)
        #expect(histogram.max == 4.0.nextUp)
    }

    @Test
    func handlesOneDistinctValue() {
        let histogram = Histogram(values: [2, 2, 2], buckets: 3)
        #expect(histogram.sum == 3)
        #expect(histogram.buckets.reduce(0, +) == 3)
    }
}
