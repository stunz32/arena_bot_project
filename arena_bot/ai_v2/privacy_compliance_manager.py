"""
Privacy Compliance Manager - Data Privacy Controls and API Usage Compliance

Provides comprehensive data privacy controls, API usage compliance monitoring,
user consent management, and data protection measures for Arena Bot AI v2.
"""

import logging
import json
import hashlib
from typing import Dict, List, Any, Optional, Set, Tuple
from datetime import datetime, timedelta
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum
import threading
import uuid


class DataCategory(Enum):
    """Categories of data being processed."""
    PERSONAL = "personal"
    GAMEPLAY = "gameplay"
    PREFERENCES = "preferences"
    PERFORMANCE = "performance"
    ANALYTICS = "analytics"
    CACHED = "cached"
    TEMPORARY = "temporary"


class ConsentStatus(Enum):
    """User consent status."""
    GRANTED = "granted"
    DENIED = "denied"
    PENDING = "pending"
    WITHDRAWN = "withdrawn"
    EXPIRED = "expired"


class DataProcessingPurpose(Enum):
    """Purposes for data processing."""
    DRAFT_ASSISTANCE = "draft_assistance"
    PERFORMANCE_ANALYSIS = "performance_analysis"
    PERSONALIZATION = "personalization"
    SYSTEM_OPTIMIZATION = "system_optimization"
    ERROR_REPORTING = "error_reporting"
    ANALYTICS = "analytics"


@dataclass
class ConsentRecord:
    """Record of user consent."""
    user_id: str
    data_category: DataCategory
    purpose: DataProcessingPurpose
    status: ConsentStatus
    granted_at: Optional[datetime] = None
    withdrawn_at: Optional[datetime] = None
    expires_at: Optional[datetime] = None
    version: str = "1.0"
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None


@dataclass
class DataProcessingLog:
    """Log entry for data processing activities."""
    timestamp: datetime
    user_id: str
    data_category: DataCategory
    purpose: DataProcessingPurpose
    operation: str
    data_description: str
    legal_basis: str
    retention_period: Optional[timedelta] = None
    anonymized: bool = False


@dataclass
class APIUsageRecord:
    """Record of API usage for compliance monitoring."""
    timestamp: datetime
    api_endpoint: str
    request_type: str
    data_sent: str
    response_received: bool
    user_consent_verified: bool
    rate_limit_respected: bool
    data_minimization_applied: bool


class PrivacyComplianceManager:
    """
    Comprehensive privacy and compliance management system.
    
    Handles user consent, data processing transparency, API usage compliance,
    and data protection measures in accordance with privacy regulations.
    """
    
    def __init__(self, compliance_config_file: Optional[str] = None):
        """Initialize privacy compliance manager."""
        self.logger = logging.getLogger(__name__)
        
        # Configuration
        self.config_file = Path(compliance_config_file) if compliance_config_file else Path("config/privacy_config.json")
        self.config_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Storage paths
        self.consent_storage = Path("privacy_data/consent_records.json")
        self.processing_log_storage = Path("privacy_data/processing_logs.json")
        self.api_usage_storage = Path("privacy_data/api_usage_logs.json")
        
        # Ensure storage directories exist
        for storage_path in [self.consent_storage, self.processing_log_storage, self.api_usage_storage]:
            storage_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Data structures
        self.consent_records = {}  # user_id -> List[ConsentRecord]
        self.processing_logs = []
        self.api_usage_logs = []
        
        # Thread safety
        self.consent_lock = threading.Lock()
        self.processing_lock = threading.Lock()
        self.api_usage_lock = threading.Lock()
        
        # Compliance configuration
        self.compliance_config = self._load_compliance_config()
        
        # Data retention policies (in days)
        self.retention_policies = {
            DataCategory.PERSONAL: 365,        # 1 year
            DataCategory.GAMEPLAY: 180,        # 6 months
            DataCategory.PREFERENCES: 730,     # 2 years
            DataCategory.PERFORMANCE: 90,      # 3 months
            DataCategory.ANALYTICS: 30,        # 1 month
            DataCategory.CACHED: 7,            # 1 week
            DataCategory.TEMPORARY: 1          # 1 day
        }
        
        # API compliance settings
        self.api_compliance_settings = {
            'hsreplay_api': {
                'rate_limit_per_minute': 60,
                'rate_limit_per_hour': 1000,
                'requires_user_consent': True,
                'data_minimization': True,
                'allowed_purposes': [
                    DataProcessingPurpose.DRAFT_ASSISTANCE,
                    DataProcessingPurpose.PERFORMANCE_ANALYSIS
                ]
            }
        }
        
        # Load existing data
        self._load_stored_data()
        
        self.logger.info("PrivacyComplianceManager initialized")
    
    def request_consent(self, user_id: str, data_category: DataCategory, 
                       purpose: DataProcessingPurpose, 
                       metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Request user consent for data processing."""
        try:
            with self.consent_lock:
                consent_record = ConsentRecord(
                    user_id=user_id,
                    data_category=data_category,
                    purpose=purpose,
                    status=ConsentStatus.PENDING,
                    version=self.compliance_config['consent_version'],
                    ip_address=metadata.get('ip_address') if metadata else None,
                    user_agent=metadata.get('user_agent') if metadata else None
                )
                
                # Store consent request
                if user_id not in self.consent_records:
                    self.consent_records[user_id] = []
                
                self.consent_records[user_id].append(consent_record)
                
                # For this implementation, we'll auto-grant consent
                # In a real implementation, this would trigger a UI prompt
                self._grant_consent(user_id, data_category, purpose)
                
                self._save_consent_records()
                
                self.logger.info(f"Consent requested for user {user_id}: {data_category.value} - {purpose.value}")
                return True
                
        except Exception as e:
            self.logger.error(f"Error requesting consent: {e}")
            return False
    
    def grant_consent(self, user_id: str, data_category: DataCategory, 
                     purpose: DataProcessingPurpose) -> bool:
        """Grant consent for data processing."""
        return self._grant_consent(user_id, data_category, purpose)
    
    def withdraw_consent(self, user_id: str, data_category: DataCategory, 
                        purpose: DataProcessingPurpose) -> bool:
        """Withdraw consent for data processing."""
        try:
            with self.consent_lock:
                if user_id in self.consent_records:
                    for record in self.consent_records[user_id]:
                        if (record.data_category == data_category and 
                            record.purpose == purpose and 
                            record.status == ConsentStatus.GRANTED):
                            
                            record.status = ConsentStatus.WITHDRAWN
                            record.withdrawn_at = datetime.now()
                            
                            # Trigger data deletion if required
                            self._handle_consent_withdrawal(user_id, data_category, purpose)
                            
                            self._save_consent_records()
                            
                            self.logger.info(f"Consent withdrawn for user {user_id}: {data_category.value} - {purpose.value}")
                            return True
                
                return False
                
        except Exception as e:
            self.logger.error(f"Error withdrawing consent: {e}")
            return False
    
    def check_consent(self, user_id: str, data_category: DataCategory, 
                     purpose: DataProcessingPurpose) -> bool:
        """Check if user has granted consent for specific data processing."""
        try:
            with self.consent_lock:
                if user_id not in self.consent_records:
                    return False
                
                for record in self.consent_records[user_id]:
                    if (record.data_category == data_category and 
                        record.purpose == purpose and 
                        record.status == ConsentStatus.GRANTED):
                        
                        # Check if consent has expired
                        if record.expires_at and datetime.now() > record.expires_at:
                            record.status = ConsentStatus.EXPIRED
                            self._save_consent_records()
                            return False
                        
                        return True
                
                return False
                
        except Exception as e:
            self.logger.error(f"Error checking consent: {e}")
            return False
    
    def log_data_processing(self, user_id: str, data_category: DataCategory,
                          purpose: DataProcessingPurpose, operation: str,
                          data_description: str, legal_basis: str = "consent",
                          anonymized: bool = False) -> bool:
        """Log data processing activity."""
        try:
            # Check consent first
            if legal_basis == "consent" and not self.check_consent(user_id, data_category, purpose):
                self.logger.warning(f"Data processing attempted without consent: {user_id} - {data_category.value}")
                return False
            
            with self.processing_lock:
                log_entry = DataProcessingLog(
                    timestamp=datetime.now(),
                    user_id=user_id,
                    data_category=data_category,
                    purpose=purpose,
                    operation=operation,
                    data_description=data_description,
                    legal_basis=legal_basis,
                    retention_period=timedelta(days=self.retention_policies[data_category]),
                    anonymized=anonymized
                )
                
                self.processing_logs.append(log_entry)
                
                # Periodic cleanup to manage log size
                if len(self.processing_logs) % 100 == 0:
                    self._cleanup_old_logs()
                
                self._save_processing_logs()
                
                return True
                
        except Exception as e:
            self.logger.error(f"Error logging data processing: {e}")
            return False
    
    def log_api_usage(self, api_endpoint: str, request_type: str, data_sent: str,
                     user_id: Optional[str] = None, response_received: bool = True) -> bool:
        """Log API usage for compliance monitoring."""
        try:
            with self.api_usage_lock:
                # Check user consent if user_id provided
                user_consent_verified = True
                if user_id:
                    user_consent_verified = self.check_consent(
                        user_id, DataCategory.GAMEPLAY, DataProcessingPurpose.DRAFT_ASSISTANCE
                    )
                
                # Check rate limiting compliance
                rate_limit_respected = self._check_api_rate_limit(api_endpoint)
                
                # Apply data minimization
                data_minimization_applied = self._apply_data_minimization(data_sent)
                
                usage_record = APIUsageRecord(
                    timestamp=datetime.now(),
                    api_endpoint=api_endpoint,
                    request_type=request_type,
                    data_sent=self._hash_sensitive_data(data_sent),
                    response_received=response_received,
                    user_consent_verified=user_consent_verified,
                    rate_limit_respected=rate_limit_respected,
                    data_minimization_applied=data_minimization_applied
                )
                
                self.api_usage_logs.append(usage_record)
                
                # Periodic cleanup
                if len(self.api_usage_logs) % 50 == 0:
                    self._cleanup_old_api_logs()
                
                self._save_api_usage_logs()
                
                return user_consent_verified and rate_limit_respected
                
        except Exception as e:
            self.logger.error(f"Error logging API usage: {e}")
            return False
    
    def anonymize_user_data(self, user_id: str) -> bool:
        """Anonymize all data for a specific user."""
        try:
            anonymized_id = self._generate_anonymous_id(user_id)
            
            # Anonymize consent records
            with self.consent_lock:
                if user_id in self.consent_records:
                    self.consent_records[anonymized_id] = self.consent_records.pop(user_id)
                    for record in self.consent_records[anonymized_id]:
                        record.user_id = anonymized_id
            
            # Anonymize processing logs
            with self.processing_lock:
                for log in self.processing_logs:
                    if log.user_id == user_id:
                        log.user_id = anonymized_id
                        log.anonymized = True
            
            self._save_all_data()
            
            self.logger.info(f"Anonymized data for user: {user_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error anonymizing user data: {e}")
            return False
    
    def delete_user_data(self, user_id: str, data_category: Optional[DataCategory] = None) -> bool:
        """Delete user data (right to erasure)."""
        try:
            # Delete consent records
            with self.consent_lock:
                if user_id in self.consent_records:
                    if data_category:
                        # Delete specific category
                        self.consent_records[user_id] = [
                            record for record in self.consent_records[user_id]
                            if record.data_category != data_category
                        ]
                        if not self.consent_records[user_id]:
                            del self.consent_records[user_id]
                    else:
                        # Delete all consent records
                        del self.consent_records[user_id]
            
            # Delete processing logs
            with self.processing_lock:
                if data_category:
                    self.processing_logs = [
                        log for log in self.processing_logs
                        if not (log.user_id == user_id and log.data_category == data_category)
                    ]
                else:
                    self.processing_logs = [
                        log for log in self.processing_logs if log.user_id != user_id
                    ]
            
            self._save_all_data()
            
            category_str = f" ({data_category.value})" if data_category else ""
            self.logger.info(f"Deleted user data{category_str}: {user_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error deleting user data: {e}")
            return False
    
    def export_user_data(self, user_id: str) -> Dict[str, Any]:
        """Export all user data (data portability)."""
        try:
            user_data = {
                'user_id': user_id,
                'export_timestamp': datetime.now().isoformat(),
                'consent_records': [],
                'processing_logs': [],
                'api_usage_logs': []
            }
            
            # Export consent records
            with self.consent_lock:
                if user_id in self.consent_records:
                    for record in self.consent_records[user_id]:
                        user_data['consent_records'].append({
                            'data_category': record.data_category.value,
                            'purpose': record.purpose.value,
                            'status': record.status.value,
                            'granted_at': record.granted_at.isoformat() if record.granted_at else None,
                            'withdrawn_at': record.withdrawn_at.isoformat() if record.withdrawn_at else None,
                            'version': record.version
                        })
            
            # Export processing logs (last 90 days)
            cutoff_date = datetime.now() - timedelta(days=90)
            with self.processing_lock:
                for log in self.processing_logs:
                    if log.user_id == user_id and log.timestamp > cutoff_date:
                        user_data['processing_logs'].append({
                            'timestamp': log.timestamp.isoformat(),
                            'data_category': log.data_category.value,
                            'purpose': log.purpose.value,
                            'operation': log.operation,
                            'data_description': log.data_description,
                            'legal_basis': log.legal_basis
                        })
            
            # Export API usage logs (last 30 days)
            cutoff_date = datetime.now() - timedelta(days=30)
            with self.api_usage_lock:
                for log in self.api_usage_logs:
                    # Note: API logs don't directly contain user_id, would need correlation
                    pass
            
            return user_data
            
        except Exception as e:
            self.logger.error(f"Error exporting user data: {e}")
            return {'error': str(e)}
    
    def get_compliance_report(self) -> Dict[str, Any]:
        """Generate compliance report."""
        try:
            report = {
                'report_timestamp': datetime.now().isoformat(),
                'consent_summary': self._get_consent_summary(),
                'data_processing_summary': self._get_processing_summary(),
                'api_usage_summary': self._get_api_usage_summary(),
                'compliance_status': self._assess_compliance_status(),
                'recommendations': self._generate_compliance_recommendations()
            }
            
            return report
            
        except Exception as e:
            self.logger.error(f"Error generating compliance report: {e}")
            return {'error': str(e)}
    
    # === INTERNAL METHODS ===
    
    def _load_compliance_config(self) -> Dict[str, Any]:
        """Load compliance configuration."""
        default_config = {
            'consent_version': '1.0',
            'data_retention_enabled': True,
            'anonymization_enabled': True,
            'audit_logging_enabled': True,
            'api_monitoring_enabled': True
        }
        
        try:
            if self.config_file.exists():
                with open(self.config_file, 'r') as f:
                    config = json.load(f)
                    # Merge with defaults
                    default_config.update(config)
            else:
                # Save default config
                with open(self.config_file, 'w') as f:
                    json.dump(default_config, f, indent=2)
        except Exception as e:
            self.logger.error(f"Error loading compliance config: {e}")
        
        return default_config
    
    def _grant_consent(self, user_id: str, data_category: DataCategory, 
                      purpose: DataProcessingPurpose) -> bool:
        """Internal method to grant consent."""
        try:
            with self.consent_lock:
                if user_id not in self.consent_records:
                    self.consent_records[user_id] = []
                
                # Find existing pending consent
                for record in self.consent_records[user_id]:
                    if (record.data_category == data_category and 
                        record.purpose == purpose and 
                        record.status == ConsentStatus.PENDING):
                        
                        record.status = ConsentStatus.GRANTED
                        record.granted_at = datetime.now()
                        # Set expiration (1 year from now)
                        record.expires_at = datetime.now() + timedelta(days=365)
                        return True
                
                # Create new consent record if none found
                consent_record = ConsentRecord(
                    user_id=user_id,
                    data_category=data_category,
                    purpose=purpose,
                    status=ConsentStatus.GRANTED,
                    granted_at=datetime.now(),
                    expires_at=datetime.now() + timedelta(days=365),
                    version=self.compliance_config['consent_version']
                )
                
                self.consent_records[user_id].append(consent_record)
                return True
                
        except Exception as e:
            self.logger.error(f"Error granting consent: {e}")
            return False
    
    def _handle_consent_withdrawal(self, user_id: str, data_category: DataCategory, 
                                 purpose: DataProcessingPurpose) -> None:
        """Handle actions when consent is withdrawn."""
        try:
            # Delete related cached data
            self.logger.info(f"Handling consent withdrawal: {user_id} - {data_category.value}")
            
            # This would trigger deletion of cached user data
            # Implementation depends on cache structure
            
        except Exception as e:
            self.logger.error(f"Error handling consent withdrawal: {e}")
    
    def _check_api_rate_limit(self, api_endpoint: str) -> bool:
        """Check if API rate limits are being respected."""
        try:
            # Count recent API calls to this endpoint
            current_time = datetime.now()
            minute_ago = current_time - timedelta(minutes=1)
            hour_ago = current_time - timedelta(hours=1)
            
            recent_calls_minute = sum(
                1 for log in self.api_usage_logs
                if log.api_endpoint == api_endpoint and log.timestamp > minute_ago
            )
            
            recent_calls_hour = sum(
                1 for log in self.api_usage_logs
                if log.api_endpoint == api_endpoint and log.timestamp > hour_ago
            )
            
            # Check against limits
            settings = self.api_compliance_settings.get(api_endpoint, {})
            minute_limit = settings.get('rate_limit_per_minute', 60)
            hour_limit = settings.get('rate_limit_per_hour', 1000)
            
            return recent_calls_minute < minute_limit and recent_calls_hour < hour_limit
            
        except Exception as e:
            self.logger.error(f"Error checking API rate limit: {e}")
            return True  # Default to allowing
    
    def _apply_data_minimization(self, data: str) -> bool:
        """Apply data minimization principles."""
        try:
            # Check if data contains unnecessary personal information
            # This is a simplified implementation
            sensitive_patterns = ['email', 'password', 'ssn', 'credit_card']
            
            for pattern in sensitive_patterns:
                if pattern in data.lower():
                    self.logger.warning(f"Potential sensitive data detected: {pattern}")
                    return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error applying data minimization: {e}")
            return False
    
    def _hash_sensitive_data(self, data: str) -> str:
        """Hash sensitive data for logging."""
        try:
            # Hash the data for privacy
            return hashlib.sha256(data.encode()).hexdigest()[:16]
        except Exception:
            return "hashed_data"
    
    def _generate_anonymous_id(self, user_id: str) -> str:
        """Generate anonymous ID for a user."""
        # Create deterministic anonymous ID
        salt = "arena_bot_anonymization_salt"
        combined = f"{user_id}_{salt}"
        return f"anon_{hashlib.sha256(combined.encode()).hexdigest()[:16]}"
    
    def _cleanup_old_logs(self) -> None:
        """Clean up old processing logs."""
        try:
            cutoff_date = datetime.now() - timedelta(days=365)  # Keep 1 year
            
            initial_count = len(self.processing_logs)
            self.processing_logs = [
                log for log in self.processing_logs if log.timestamp > cutoff_date
            ]
            
            cleaned_count = initial_count - len(self.processing_logs)
            if cleaned_count > 0:
                self.logger.info(f"Cleaned up {cleaned_count} old processing logs")
                
        except Exception as e:
            self.logger.error(f"Error cleaning up old logs: {e}")
    
    def _cleanup_old_api_logs(self) -> None:
        """Clean up old API usage logs."""
        try:
            cutoff_date = datetime.now() - timedelta(days=90)  # Keep 3 months
            
            initial_count = len(self.api_usage_logs)
            self.api_usage_logs = [
                log for log in self.api_usage_logs if log.timestamp > cutoff_date
            ]
            
            cleaned_count = initial_count - len(self.api_usage_logs)
            if cleaned_count > 0:
                self.logger.info(f"Cleaned up {cleaned_count} old API usage logs")
                
        except Exception as e:
            self.logger.error(f"Error cleaning up old API logs: {e}")
    
    def _load_stored_data(self) -> None:
        """Load stored consent and log data."""
        try:
            # Load consent records
            if self.consent_storage.exists():
                with open(self.consent_storage, 'r') as f:
                    data = json.load(f)
                    for user_id, records in data.items():
                        self.consent_records[user_id] = [
                            self._dict_to_consent_record(record) for record in records
                        ]
            
            # Load processing logs
            if self.processing_log_storage.exists():
                with open(self.processing_log_storage, 'r') as f:
                    data = json.load(f)
                    self.processing_logs = [
                        self._dict_to_processing_log(log) for log in data
                    ]
            
            # Load API usage logs
            if self.api_usage_storage.exists():
                with open(self.api_usage_storage, 'r') as f:
                    data = json.load(f)
                    self.api_usage_logs = [
                        self._dict_to_api_usage_record(log) for log in data
                    ]
                    
        except Exception as e:
            self.logger.error(f"Error loading stored data: {e}")
    
    def _save_consent_records(self) -> None:
        """Save consent records to storage."""
        try:
            data = {}
            for user_id, records in self.consent_records.items():
                data[user_id] = [self._consent_record_to_dict(record) for record in records]
            
            with open(self.consent_storage, 'w') as f:
                json.dump(data, f, indent=2, default=str)
                
        except Exception as e:
            self.logger.error(f"Error saving consent records: {e}")
    
    def _save_processing_logs(self) -> None:
        """Save processing logs to storage."""
        try:
            data = [self._processing_log_to_dict(log) for log in self.processing_logs]
            
            with open(self.processing_log_storage, 'w') as f:
                json.dump(data, f, indent=2, default=str)
                
        except Exception as e:
            self.logger.error(f"Error saving processing logs: {e}")
    
    def _save_api_usage_logs(self) -> None:
        """Save API usage logs to storage."""
        try:
            data = [self._api_usage_record_to_dict(log) for log in self.api_usage_logs]
            
            with open(self.api_usage_storage, 'w') as f:
                json.dump(data, f, indent=2, default=str)
                
        except Exception as e:
            self.logger.error(f"Error saving API usage logs: {e}")
    
    def _save_all_data(self) -> None:
        """Save all data to storage."""
        self._save_consent_records()
        self._save_processing_logs()
        self._save_api_usage_logs()
    
    def _dict_to_consent_record(self, data: Dict[str, Any]) -> ConsentRecord:
        """Convert dictionary to ConsentRecord."""
        return ConsentRecord(
            user_id=data['user_id'],
            data_category=DataCategory(data['data_category']),
            purpose=DataProcessingPurpose(data['purpose']),
            status=ConsentStatus(data['status']),
            granted_at=datetime.fromisoformat(data['granted_at']) if data.get('granted_at') else None,
            withdrawn_at=datetime.fromisoformat(data['withdrawn_at']) if data.get('withdrawn_at') else None,
            expires_at=datetime.fromisoformat(data['expires_at']) if data.get('expires_at') else None,
            version=data.get('version', '1.0'),
            ip_address=data.get('ip_address'),
            user_agent=data.get('user_agent')
        )
    
    def _consent_record_to_dict(self, record: ConsentRecord) -> Dict[str, Any]:
        """Convert ConsentRecord to dictionary."""
        return {
            'user_id': record.user_id,
            'data_category': record.data_category.value,
            'purpose': record.purpose.value,
            'status': record.status.value,
            'granted_at': record.granted_at.isoformat() if record.granted_at else None,
            'withdrawn_at': record.withdrawn_at.isoformat() if record.withdrawn_at else None,
            'expires_at': record.expires_at.isoformat() if record.expires_at else None,
            'version': record.version,
            'ip_address': record.ip_address,
            'user_agent': record.user_agent
        }
    
    def _dict_to_processing_log(self, data: Dict[str, Any]) -> DataProcessingLog:
        """Convert dictionary to DataProcessingLog."""
        return DataProcessingLog(
            timestamp=datetime.fromisoformat(data['timestamp']),
            user_id=data['user_id'],
            data_category=DataCategory(data['data_category']),
            purpose=DataProcessingPurpose(data['purpose']),
            operation=data['operation'],
            data_description=data['data_description'],
            legal_basis=data['legal_basis'],
            retention_period=timedelta(days=data['retention_period_days']) if data.get('retention_period_days') else None,
            anonymized=data.get('anonymized', False)
        )
    
    def _processing_log_to_dict(self, log: DataProcessingLog) -> Dict[str, Any]:
        """Convert DataProcessingLog to dictionary."""
        return {
            'timestamp': log.timestamp.isoformat(),
            'user_id': log.user_id,
            'data_category': log.data_category.value,
            'purpose': log.purpose.value,
            'operation': log.operation,
            'data_description': log.data_description,
            'legal_basis': log.legal_basis,
            'retention_period_days': log.retention_period.days if log.retention_period else None,
            'anonymized': log.anonymized
        }
    
    def _dict_to_api_usage_record(self, data: Dict[str, Any]) -> APIUsageRecord:
        """Convert dictionary to APIUsageRecord."""
        return APIUsageRecord(
            timestamp=datetime.fromisoformat(data['timestamp']),
            api_endpoint=data['api_endpoint'],
            request_type=data['request_type'],
            data_sent=data['data_sent'],
            response_received=data['response_received'],
            user_consent_verified=data['user_consent_verified'],
            rate_limit_respected=data['rate_limit_respected'],
            data_minimization_applied=data['data_minimization_applied']
        )
    
    def _api_usage_record_to_dict(self, record: APIUsageRecord) -> Dict[str, Any]:
        """Convert APIUsageRecord to dictionary."""
        return {
            'timestamp': record.timestamp.isoformat(),
            'api_endpoint': record.api_endpoint,
            'request_type': record.request_type,
            'data_sent': record.data_sent,
            'response_received': record.response_received,
            'user_consent_verified': record.user_consent_verified,
            'rate_limit_respected': record.rate_limit_respected,
            'data_minimization_applied': record.data_minimization_applied
        }
    
    def _get_consent_summary(self) -> Dict[str, Any]:
        """Get consent summary for compliance report."""
        try:
            total_users = len(self.consent_records)
            total_consents = sum(len(records) for records in self.consent_records.values())
            
            granted_consents = sum(
                1 for records in self.consent_records.values()
                for record in records if record.status == ConsentStatus.GRANTED
            )
            
            return {
                'total_users': total_users,
                'total_consent_requests': total_consents,
                'granted_consents': granted_consents,
                'consent_rate': (granted_consents / total_consents * 100) if total_consents > 0 else 0
            }
            
        except Exception as e:
            self.logger.error(f"Error getting consent summary: {e}")
            return {}
    
    def _get_processing_summary(self) -> Dict[str, Any]:
        """Get data processing summary."""
        try:
            recent_logs = [
                log for log in self.processing_logs
                if (datetime.now() - log.timestamp).days <= 30
            ]
            
            return {
                'total_processing_activities': len(self.processing_logs),
                'recent_activities_30_days': len(recent_logs),
                'categories_processed': len(set(log.data_category for log in recent_logs)),
                'purposes_active': len(set(log.purpose for log in recent_logs))
            }
            
        except Exception as e:
            self.logger.error(f"Error getting processing summary: {e}")
            return {}
    
    def _get_api_usage_summary(self) -> Dict[str, Any]:
        """Get API usage summary."""
        try:
            recent_logs = [
                log for log in self.api_usage_logs
                if (datetime.now() - log.timestamp).days <= 30
            ]
            
            consent_violations = sum(
                1 for log in recent_logs if not log.user_consent_verified
            )
            
            rate_limit_violations = sum(
                1 for log in recent_logs if not log.rate_limit_respected
            )
            
            return {
                'total_api_calls': len(self.api_usage_logs),
                'recent_calls_30_days': len(recent_logs),
                'consent_violations': consent_violations,
                'rate_limit_violations': rate_limit_violations,
                'compliance_rate': ((len(recent_logs) - consent_violations - rate_limit_violations) / len(recent_logs) * 100) if recent_logs else 100
            }
            
        except Exception as e:
            self.logger.error(f"Error getting API usage summary: {e}")
            return {}
    
    def _assess_compliance_status(self) -> Dict[str, Any]:
        """Assess overall compliance status."""
        try:
            issues = []
            
            # Check for consent violations
            consent_summary = self._get_consent_summary()
            if consent_summary.get('consent_rate', 0) < 90:
                issues.append("Low consent rate")
            
            # Check for API violations
            api_summary = self._get_api_usage_summary()
            if api_summary.get('compliance_rate', 0) < 95:
                issues.append("API compliance issues")
            
            # Check data retention
            old_logs = [
                log for log in self.processing_logs
                if (datetime.now() - log.timestamp).days > 365
            ]
            if old_logs:
                issues.append("Old data not properly cleaned")
            
            status = "compliant" if not issues else "non_compliant"
            
            return {
                'status': status,
                'issues': issues,
                'last_assessment': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error assessing compliance status: {e}")
            return {'status': 'unknown', 'error': str(e)}
    
    def _generate_compliance_recommendations(self) -> List[str]:
        """Generate compliance recommendations."""
        recommendations = []
        
        try:
            compliance_status = self._assess_compliance_status()
            
            if "Low consent rate" in compliance_status.get('issues', []):
                recommendations.append("Improve consent request process and user experience")
            
            if "API compliance issues" in compliance_status.get('issues', []):
                recommendations.append("Review API usage patterns and implement better rate limiting")
            
            if "Old data not properly cleaned" in compliance_status.get('issues', []):
                recommendations.append("Implement automated data retention and cleanup policies")
            
            if not recommendations:
                recommendations.append("Compliance status is good, continue monitoring")
            
        except Exception as e:
            self.logger.error(f"Error generating recommendations: {e}")
            recommendations.append("Error generating recommendations")
        
        return recommendations


# Global privacy compliance manager instance
_privacy_manager = None

def get_privacy_manager() -> PrivacyComplianceManager:
    """Get global privacy compliance manager instance."""
    global _privacy_manager
    if _privacy_manager is None:
        _privacy_manager = PrivacyComplianceManager()
    return _privacy_manager