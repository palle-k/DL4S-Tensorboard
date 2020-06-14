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


public struct Histogram {
    public var buckets: [Double]
    public var min: Double
    public var max: Double
    
    public var sum: Double {
        return buckets.reduce(0, +)
    }
    
    public var edges: [Double] {
        (0 ... buckets.count).map { i -> Double in
            Double(i) / Double(buckets.count) * (max - min) + min
        }
    }
    
    public init(values: [Double], buckets numBuckets: Int) {
        buckets = Array(repeating: 0, count: numBuckets)
        min = values.reduce(Double.infinity, Double.minimum)
        max = values.reduce(-Double.infinity, Double.maximum).nextUp
        
        for el in values {
            let bucketIdx = Swift.min(Int((el - min) / (max - min) * Double(numBuckets)), numBuckets - 1)
            buckets[bucketIdx] += 1
        }
    }
}
