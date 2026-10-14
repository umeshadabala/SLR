class ApiConfig {
  static String _baseUrl = 'http://127.0.0.1:8000';
  static Duration timeout = const Duration(seconds: 10);
  static Duration telemetryPollInterval = const Duration(milliseconds: 1500);

  static String get baseUrl => _baseUrl;

  static void setBaseUrl(String url) {
    String trimmed = url.trim();
    if (trimmed.endsWith('/')) {
      trimmed = trimmed.substring(0, trimmed.length - 1);
    }
    _baseUrl = trimmed;
  }
}
