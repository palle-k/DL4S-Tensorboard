//
//  Histogram.swift
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

/// A histogram with buckets of equal width between the smallest and the largest value.
public struct Histogram: Sendable, Equatable {
    /// Number of values in each bucket.
    public var buckets: [Double]

    /// Lower edge of the first bucket.
    public var min: Double

    /// Upper edge of the last bucket. It is the value after the largest value, so the largest value falls into the last bucket.
    public var max: Double

    /// Total number of values.
    public var sum: Double {
        buckets.reduce(0, +)
    }

    /// The `buckets.count + 1` bucket edges from `min` to `max`.
    public var edges: [Double] {
        (0 ... buckets.count).map { i -> Double in
            Double(i) / Double(buckets.count) * (max - min) + min
        }
    }

    /// Counts the values into buckets of equal width.
    /// - Parameters:
    ///   - values: Values to count. Must not be empty.
    ///   - numBuckets: Number of buckets. Must be at least 1.
    public init(values: [Double], buckets numBuckets: Int) {
        precondition(!values.isEmpty, "Histogram needs at least one value.")
        precondition(numBuckets >= 1, "Histogram needs at least one bucket.")
        buckets = Array(repeating: 0, count: numBuckets)
        min = values.reduce(Double.infinity, Double.minimum)
        max = values.reduce(-Double.infinity, Double.maximum).nextUp

        for el in values {
            let bucketIdx = Swift.min(Int((el - min) / (max - min) * Double(numBuckets)), numBuckets - 1)
            buckets[bucketIdx] += 1
        }
    }
}
