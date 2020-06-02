//
//  File.swift
//  
//
//  Created by Palle Klewitz on 02.06.20.
//

import Foundation
import XCTest
import DL4S
@testable import DL4STensorboard

class HistogramTests: XCTestCase {
    func testHistogramCreation() {
        let t = Tensor<Float, CPU>(normalDistributedWithShape: 100)
        let histogram = Histogram(values: t.elements.map(Double.init), buckets: 20)
        print(histogram)
    }
}
