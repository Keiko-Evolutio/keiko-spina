# Contributing to Keiko Backend

Thank you for your interest in contributing to Keiko Backend! This guide will help you get started.

## 🚀 Quick Start

### Prerequisites
- Python 3.11 or 3.12
- [uv](https://docs.astral.sh/uv/) (recommended) or pip
- Git
- Docker & Docker Compose (for full development environment)

### Development Setup
```bash
# Clone the repository
git clone https://github.com/Keiko-Development/keiko-backbone.git
cd keiko-backbone

# Set up development environment
make dev

# Install dependencies
cd backend
uv sync --group dev --group test

# Run tests
make test
```

## 🛠 Development Workflow

### 1. Before You Start
- Check [existing issues](https://github.com/Keiko-Development/keiko-backbone/issues) for similar work
- For major changes, create an issue to discuss the approach
- Fork the repository and create a feature branch

### 2. Making Changes
```bash
# Create a new branch
git checkout -b feature/your-feature-name

# Make your changes
# ... code, code, code ...

# Run quality checks
make lint        # Ruff linting and formatting  
make type-check  # MyPy type checking
make test        # Run test suite
make quality     # Run all quality checks
```

### 3. Commit Guidelines
We follow [Conventional Commits](https://www.conventionalcommits.org/):

```bash
# Format: type(scope): description
git commit -m "feat(api): add user authentication endpoint"
git commit -m "fix(auth): resolve token validation bug"
git commit -m "docs(readme): update installation instructions"
```

**Types:**
- `feat`: New features
- `fix`: Bug fixes  
- `docs`: Documentation changes
- `style`: Code formatting (no logic changes)
- `refactor`: Code restructuring
- `test`: Adding or updating tests
- `chore`: Build process, dependencies, etc.

### 4. Pull Request Process
1. **Update your branch** with the latest main
2. **Run all quality checks** locally
3. **Create a Pull Request** using our template
4. **Address review feedback** promptly
5. **Ensure CI passes** before merge

## 📋 Coding Standards

### Python Code Style
- **Formatter**: Ruff (replaces Black)
- **Linter**: Ruff (replaces flake8, isort)  
- **Type Checker**: MyPy (strict mode)
- **Line Length**: 88 characters
- **Import Sorting**: Automatic via Ruff

### Code Quality Requirements
- **Test Coverage**: Minimum 85%
- **Type Hints**: Required for all new code
- **Documentation**: Docstrings for public APIs
- **Security**: No hardcoded secrets, proper input validation

### Project Structure
```
backend/
├── api/           # FastAPI routes and endpoints
├── core/          # Core utilities and exceptions  
├── data_models/   # Pydantic models
├── services/      # Business logic
├── auth/          # Authentication & authorization
├── monitoring/    # Observability and metrics
└── tests/         # Test suites
    ├── unit/      # Unit tests
    ├── integration/  # Integration tests
    └── e2e/       # End-to-end tests
```

## 🧪 Testing

### Test Categories
- **Unit Tests**: Fast, isolated component tests
- **Integration Tests**: Component interaction tests
- **E2E Tests**: Full workflow tests
- **Performance Tests**: Load and benchmark tests

### Running Tests
```bash
# All tests
make test

# Specific test types
pytest tests/unit/           # Unit tests only
pytest tests/integration/    # Integration tests  
pytest tests/e2e/           # E2E tests

# With coverage
make test-cov

# Fast parallel execution  
make test-fast
```

### Writing Tests
- Use descriptive test names: `test_user_authentication_with_valid_credentials`
- Follow AAA pattern: Arrange, Act, Assert
- Mock external dependencies
- Test edge cases and error conditions

## 🔒 Security

### Security Best Practices
- **Never commit secrets** to version control
- **Validate all inputs** in API endpoints
- **Use parameterized queries** for database access
- **Implement proper authentication** and authorization
- **Follow OWASP guidelines** for web application security

### Security Review
- All security-related changes require extra review
- Security vulnerabilities should be reported privately
- See [SECURITY.md](.github/SECURITY.md) for details

## 📚 Documentation

### Code Documentation
- **Docstrings**: Use Google-style docstrings
- **Type Hints**: Required for all public APIs
- **Comments**: Explain complex logic, not obvious code
- **README**: Keep installation and usage instructions current

### API Documentation
- **OpenAPI**: Automatically generated from FastAPI
- **Examples**: Include request/response examples
- **Error Handling**: Document error codes and responses

## 🚀 Performance

### Performance Guidelines
- **Async/Await**: Use async patterns for I/O operations
- **Database Queries**: Optimize N+1 queries
- **Caching**: Implement appropriate caching strategies
- **Monitoring**: Add metrics for performance-critical paths

### Performance Testing
- Benchmark performance-sensitive changes
- Monitor memory usage and response times
- Use profiling tools to identify bottlenecks

## 🌟 Recognition

### Contributors
- All contributors are acknowledged in our releases
- Significant contributors may be added to the README
- We appreciate all forms of contribution!

### Types of Contributions
- 🐛 **Bug Reports**: Help us identify and fix issues
- ✨ **Features**: New functionality and enhancements  
- 📚 **Documentation**: Improve guides and examples
- 🧪 **Testing**: Add test coverage and scenarios
- 🔍 **Code Review**: Review pull requests
- 💬 **Community**: Help in discussions and issues

## ❓ Getting Help

### Communication Channels
- **GitHub Issues**: Bug reports and feature requests
- **GitHub Discussions**: Questions and community chat
- **Email**: oscharko@keiko.dev for direct contact

### Development Questions
- Check existing issues and discussions first
- Provide minimal reproduction examples
- Include relevant system information
- Be specific about the problem or question

## 📜 License

By contributing, you agree that your contributions will be licensed under the same license as the project (see [LICENSE](LICENSE) file).

---

**Thank you for contributing to Keiko Backend!** 🎉

Your contributions help make this project better for everyone.