# TradeMentor Stock Analysis System

## Project Overview

TradeMentor is a comprehensive stock analysis system that integrates stock data analysis, technical indicator calculations, AI-powered consultation, and community forum features. The system provides professional stock analysis tools to help investors make more informed investment decisions.

## Key Features

### 🎯 Core Features
- **Stock Data Analysis**: Supports historical data retrieval and analysis for A-shares, US stocks, and Hong Kong stocks
- **Technical Indicator Calculations**: Includes MA, MACD, KDJ, RSI, and other technical indicators
- **Intelligent Scoring System**: Comprehensive stock scoring based on multi-dimensional indicators
- **Chart Visualization**: Generates professional stock analysis charts and reports
- **AI-Powered Consultation**: Integrated OpenAI intelligent trading advisor service
- **Community Forum**: Investor communication platform

### 📊 Technical Features
- **Multi-Market Support**: Comprehensive coverage of A-shares, US stocks, and Hong Kong stocks
- **Real-time Data**: Real-time stock data based on AkShare
- **High Performance**: Supports batch stock analysis with multi-process processing
- **Visualization**: Professional charts generated using Matplotlib and Plotly
- **RESTful API**: Complete API interface services

## System Requirements

- **Python Version**: 3.10.18
- **Operating System**: Windows/Linux/macOS
- **Memory**: Recommended 4GB or more
- **Network**: Stable internet connection required for stock data retrieval

## Installation Guide

### 1. Environment Setup

Ensure your system has Python 3.10.18 installed. We recommend using Conda for environment management:

```bash
# Create a new conda environment
conda create -n tradementor python=3.10.18
conda activate tradementor
```

### 2. Install Dependencies

Execute in the project root directory:

```bash
# Install all required Python packages
pip install -r requirements.txt
```

### 3. Environment Configuration

Create configuration file (optional):

```bash
# Copy configuration file template
cp config.example.toml config.toml
# Edit configuration file as needed
```

## Usage Instructions

### Starting Main Services

1. **Start API Service**:
   ```bash
   python api.py
   ```
   The service will start on the default port, providing stock analysis API interfaces.

2. **Start AI Consultation System**:
   ```bash
   # Switch to chatbox directory
   cd chatbox
   # Start TradeMentor service
   python TradeMentor.py
   ```
   The AI consultation system will start on `localhost:8001`.

### Accessing the System

- **Main Service**: Access the web interface provided by the API service through your browser
- **AI Consultation**: Visit `http://localhost:8001` to use the intelligent trading advisor
- **API Interface**: Programmatic calls can be made through the API interface

## Project Structure

```
TradeMentorDataSite/
├── api.py                 # Main API service
├── stock_analyze.py       # Stock analysis core module
├── database.py           # Database management
├── chatbox/              # AI consultation system
│   └── TradeMentor.py   # AI service main program
├── modules/              # Feature modules
├── static/               # Static resources
├── templates/            # Template files
├── forum/               # Forum related
├── requirements.txt      # Dependency package list
└── README.md           # Project documentation
```

## Core Module Descriptions

### stock_analyze.py
- Stock historical data retrieval
- Technical indicator calculations (MA, MACD, KDJ, RSI, etc.)
- Stock scoring algorithms
- Chart generation functionality

### api.py
- FastAPI web service
- RESTful API interfaces
- Static file service
- Forum functionality

### TradeMentor.py
- OpenAI integration
- Intelligent dialogue system
- Multiple dialogue modes (humorous, professional, educational, supportive)

## API Interfaces

### Stock Analysis APIs
- `POST /analyze/single` - Single stock analysis
- `POST /analyze/batch` - Batch stock analysis
- `GET /stocks/list` - Get stock list

### Forum APIs
- `GET /api/forum/posts` - Get post list
- `POST /api/forum/posts` - Create new post
- `GET /api/forum/categories` - Get category list

## Configuration

### Environment Variables
The AI consultation system requires the following environment variables:
- `OPENAI_API_KEY`: OpenAI API key
- `OPENAI_ORGANIZATION`: OpenAI organization ID (optional)
- `OPENAI_BASE_URL`: OpenAI API base URL (optional)
- `MODEL_NAME`: Model name to use

### Configuration File
Detailed configuration can be done through the `config.toml` file:
- Database connections
- API keys
- Service ports
- Cache settings

## Development Guide

### Adding New Technical Indicators
Add new indicator calculation logic in the `calculate_technical_indicators` function in `stock_analyze.py`.

### Extending AI Features
Add new dialogue modes and processing logic in `chatbox/TradeMentor.py`.

### Customizing Scoring Algorithms
Modify the `calculate_stock_score` function to adjust stock scoring algorithms.

## Troubleshooting

### Common Issues

1. **Data Retrieval Failure**
   - Check network connection
   - Confirm stock code format is correct
   - View log files for detailed error information

2. **AI Service Cannot Start**
   - Confirm environment variables are configured correctly
   - Check OpenAI API key validity
   - Verify network connection

3. **Chart Generation Failure**
   - Check matplotlib font configuration
   - Confirm output directory permissions
   - Check memory usage

### Log Files
- Main service logs: Check console output
- AI service logs: `chatbox/TradeMentor.log`

## License

This project uses an open-source license. See the [LICENSE](LICENSE) file for details.

## Contributing

We welcome Issue submissions and Pull Requests to improve the project.

## Contact

For questions or suggestions, please contact us through:
- Submit GitHub Issues
- Send emails to project maintainers

---

**Note**: The analysis results provided by this system are for reference only and do not constitute investment advice. Investment involves risks, and market entry requires caution.
