"""Version information for CustomKB."""

# Version information
VERSION_MAJOR = 0
VERSION_MINOR = 8
VERSION_PATCH = 0
VERSION_BUILD = 8

# Formatted version string
VERSION = f"{VERSION_MAJOR}.{VERSION_MINOR}.{VERSION_PATCH}"
VERSION_FULL = f"{VERSION}.{VERSION_BUILD}"

# Release info
RELEASE_DATE = "2025-07-12"

def get_version(build=False):
  """Return version string with optional build number."""
  return VERSION_FULL if build else VERSION

#fin