# Django settings overlay mounted into cvat_server at /home/django/settings.py
# Used because upstream CVAT has no env-var path for CSRF_TRUSTED_ORIGINS yet
# (see https://github.com/cvat-ai/cvat/pull/10283 — open since 2026-02).
#
# Pattern follows CVAT's officially-documented LDAP overlay:
#   docs/administration/community/advanced/ldap.md
from cvat.settings.production import *  # noqa: F401,F403

CSRF_TRUSTED_ORIGINS = ["https://cvat.jaehho.com"]
