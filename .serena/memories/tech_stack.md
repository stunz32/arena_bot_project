# Tech Stack & Dependencies

## Core Technology Stack
- **Language**: Python 3.8+ (tested with 3.11, 3.12)
- **GUI Framework**: PyQt6 (6.6.0) - Modern Qt6-based interface
- **Computer Vision**: OpenCV (4.8.1.78) for image processing
- **Image Processing**: Pillow (10.0.0) for screenshot handling
- **Numerical Computing**: NumPy (1.24.3) for matrix operations

## Secondary Dependencies
- **HTTP Requests**: requests (2.31.0), urllib3 (2.0.7)
- **Data Validation**: jsonschema (4.19.2)
- **Utilities**: python-dateutil (2.8.2), packaging (23.2)

## Development & Testing
- **Testing Framework**: pytest (7.4.3) with pytest-cov (4.1.0)
- **Test Coverage**: Comprehensive test suite with multiple scenarios
- **Performance Testing**: Stress tests and memory leak detection

## Optional/Future Dependencies
- **ML Integration**: TensorFlow (2.15.0), ONNX Runtime (1.16.0) - commented out
- **Platform Support**: Cross-platform Windows/Linux compatibility

## Key Architecture Components
- **Detection Engine**: OpenCV-based card recognition
- **Database System**: JSON-based card database with filtering
- **Logging System**: Custom async logging with TOML configuration
- **GUI System**: PyQt6-based modern interface
- **Asset Management**: Efficient loading and caching of templates