//
//  File.swift
//  
//
//  Created by Palle Klewitz on 02.06.20.
//

import Foundation
import XCTest
@testable import DL4STensorboard

class HistogramTests: XCTestCase {
    func testHistogramCreation() {
        let elements = (0 ..< 100).map {_ in Float.random(in: 0 ... 1)}
        let histogram = Histogram(values: elements.map(Double.init), buckets: 20)
        print(histogram)
    }
}
