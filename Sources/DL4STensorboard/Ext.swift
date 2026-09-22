//
//  Ext.swift
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
import SwiftGD

extension Data {
    /// The bytes of the integer in little-endian order, which the TFRecord format requires.
    init<Integer: FixedWidthInteger>(integer: Integer) {
        var value = integer.littleEndian
        self = Swift.withUnsafeBytes(of: &value) { Data($0) }
    }
}

/// Removes leading slashes and replaces characters that TensorBoard does not accept in tags.
func cleanTag(_ tag: String) -> String {
    tag
        .drop(while: { $0 == "/" })
        .replacingOccurrences(of: #"[^-/\w\.]"#, with: "_", options: .regularExpression)
}

extension FileHandle {
    func write(_ string: String) throws {
        try write(contentsOf: Data(string.utf8))
    }
}

#if canImport(DL4S)
import DL4S

extension Image {
    /// Creates an image from a tensor with shape `[height, width]` or `[channels, height, width]`.
    ///
    /// Returns `nil` if the shape is not supported or the image cannot be allocated.
    convenience init?<E: NumericType, D: DeviceType>(_ tensor: Tensor<E, D>) {
        guard 2 ... 3 ~= tensor.dim else {
            return nil
        }
        let t: Tensor<E, D>
        if tensor.dim == 3 {
            t = tensor.detached()
        } else {
            t = tensor.detached().unsqueezed(at: 0)
        }

        let (width, height) = (t.shape[2], t.shape[1])
        guard [1, 3, 4].contains(t.shape[0]) else {
            return nil
        }
        self.init(width: width, height: height)

        for y in 0 ..< height {
            for x in 0 ..< width {
                let color: Color
                let slice = t[nil, y, x]
                if slice.count == 1 {
                    color = Color(red: slice[0].item.doubleValue, green: slice[0].item.doubleValue, blue: slice[0].item.doubleValue, alpha: 1)
                } else if slice.count == 3 {
                    color = Color(red: slice[0].item.doubleValue, green: slice[1].item.doubleValue, blue: slice[2].item.doubleValue, alpha: 1)
                } else if slice.count == 4 {
                    color = Color(red: slice[0].item.doubleValue, green: slice[1].item.doubleValue, blue: slice[2].item.doubleValue, alpha: slice[3].item.doubleValue)
                } else {
                    return nil
                }
                self.set(pixel: Point(x: x, y: y), to: color)
            }
        }
    }
}

public enum TensorFlowDataTypeWrapper: Sendable {
    case float, double, int32
    
    var dtype: Tensorflow_DataType {
        switch self {
        case .float:
            return .dtFloat
        case .double:
            return .dtDouble
        case .int32:
            return .dtInt32
        }
    }
}

public struct TensorProtoWrapper: Sendable {
    var tensor: Tensorflow_TensorProto
}

public protocol TensorFlowProtoScalar: NumericType {
    static var dtype: TensorFlowDataTypeWrapper { get }
    static func populate<Device>(tensorProto: inout TensorProtoWrapper, with tensor: Tensor<Self, Device>)
}

extension Float: TensorFlowProtoScalar {
    public static var dtype: TensorFlowDataTypeWrapper {
        return .float
    }
    
    public static func populate<Device>(tensorProto: inout TensorProtoWrapper, with tensor: Tensor<Float, Device>) where Device: DeviceType {
        tensorProto.tensor.floatVal = tensor.elements
    }
}

extension Double: TensorFlowProtoScalar {
    public static var dtype: TensorFlowDataTypeWrapper {
        return .double
    }
    
    public static func populate<Device>(tensorProto: inout TensorProtoWrapper, with tensor: Tensor<Double, Device>) where Device: DeviceType {
        tensorProto.tensor.doubleVal = tensor.elements
    }
}

extension Int32: TensorFlowProtoScalar {
    public static var dtype: TensorFlowDataTypeWrapper {
        return .int32
    }
    
    public static func populate<Device>(tensorProto: inout TensorProtoWrapper, with tensor: Tensor<Int32, Device>) where Device: DeviceType {
        tensorProto.tensor.intVal = tensor.elements
    }
}
#endif
