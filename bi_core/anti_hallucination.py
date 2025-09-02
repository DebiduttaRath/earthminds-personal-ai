"""
Anti-Hallucination Guard System for Business Intelligence Platform
Implements source verification, confidence scoring, and fact-checking mechanisms
"""

import re
import json
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
import hashlib
from urllib.parse import urlparse

from bi_core.settings import settings
from bi_core.telemetry import get_logger
from bi_core.memory_optimizer import memory_optimizer

logger = get_logger(__name__)

@dataclass
class FactCheck:
    """Fact checking result"""
    claim: str
    confidence: float
    sources: List[str]
    verified: bool
    evidence: List[str]
    timestamp: datetime

@dataclass
class SourceVerification:
    """Source verification result"""
    url: str
    domain: str
    credibility_score: float
    domain_authority: str
    is_reliable: bool
    verification_notes: str

class AntiHallucinationGuard:
    """Anti-hallucination guard system for business intelligence"""
    
    def __init__(self):
        self.trusted_domains = {
            # Financial and business sources
            'sec.gov': 0.95,
            'reuters.com': 0.90,
            'bloomberg.com': 0.90,
            'wsj.com': 0.88,
            'ft.com': 0.88,
            'marketwatch.com': 0.85,
            'fool.com': 0.82,
            'yahoo.com': 0.80,
            'cnbc.com': 0.85,
            'forbes.com': 0.83,
            
            # Academic and research
            'arxiv.org': 0.92,
            'scholar.google.com': 0.88,
            'pubmed.ncbi.nlm.nih.gov': 0.90,
            'jstor.org': 0.88,
            
            # Government and regulatory
            'federalreserve.gov': 0.95,
            'treasury.gov': 0.95,
            'census.gov': 0.93,
            'bls.gov': 0.93,
            
            # Reference sources
            'wikipedia.org': 0.75,
            'investopedia.com': 0.80,
        }
        
        self.unreliable_indicators = [
            'advertisement', 'sponsored', 'affiliate', 'paid content',
            'blog.', 'personal.', 'opinion', 'editorial'
        ]
        
        # Confidence thresholds
        self.min_confidence = 0.6
        self.high_confidence = 0.8
        
    def verify_sources(self, sources: List[Dict[str, Any]]) -> List[SourceVerification]:
        """Verify the reliability of information sources"""
        verified_sources = []
        
        for source in sources:
            url = source.get('url', '')
            title = source.get('title', '')
            
            if not url:
                continue
            
            try:
                domain = urlparse(url).netloc.lower()
                
                # Remove 'www.' prefix
                if domain.startswith('www.'):
                    domain = domain[4:]
                
                # Check against trusted domains
                credibility_score = self.trusted_domains.get(domain, 0.5)
                
                # Adjust score based on URL characteristics
                if any(indicator in url.lower() for indicator in self.unreliable_indicators):
                    credibility_score -= 0.2
                
                # Check title for reliability indicators
                if any(indicator in title.lower() for indicator in self.unreliable_indicators):
                    credibility_score -= 0.1
                
                # Boost score for specific high-quality sections
                if any(section in url.lower() for section in ['investor-relations', 'sec-filings', 'financial']):
                    credibility_score += 0.1
                
                # Ensure score stays within bounds
                credibility_score = max(0.0, min(1.0, credibility_score))
                
                domain_authority = self._get_domain_authority(domain)
                is_reliable = credibility_score >= self.min_confidence
                
                verification_notes = self._generate_verification_notes(domain, credibility_score, url)
                
                verified_sources.append(SourceVerification(
                    url=url,
                    domain=domain,
                    credibility_score=credibility_score,
                    domain_authority=domain_authority,
                    is_reliable=is_reliable,
                    verification_notes=verification_notes
                ))
                
            except Exception as e:
                logger.error(f"Failed to verify source {url}: {e}")
                continue
        
        logger.info(f"Verified {len(verified_sources)} sources")
        return verified_sources
    
    def _get_domain_authority(self, domain: str) -> str:
        """Determine domain authority level"""
        if self.trusted_domains.get(domain, 0) >= 0.9:
            return "High"
        elif self.trusted_domains.get(domain, 0) >= 0.8:
            return "Medium-High"
        elif self.trusted_domains.get(domain, 0) >= 0.7:
            return "Medium"
        elif self.trusted_domains.get(domain, 0) >= 0.6:
            return "Medium-Low"
        else:
            return "Low"
    
    def _generate_verification_notes(self, domain: str, score: float, url: str) -> str:
        """Generate verification notes for a source"""
        notes = []
        
        if domain in self.trusted_domains:
            notes.append(f"Recognized trusted source ({domain})")
        
        if score >= 0.9:
            notes.append("Highly credible source")
        elif score >= 0.8:
            notes.append("Very credible source")
        elif score >= 0.7:
            notes.append("Credible source")
        elif score >= 0.6:
            notes.append("Moderately credible source")
        else:
            notes.append("Lower credibility source - verify independently")
        
        if any(indicator in url.lower() for indicator in self.unreliable_indicators):
            notes.append("Contains potential bias indicators")
        
        return "; ".join(notes)
    
    def check_numerical_claims(self, text: str) -> List[FactCheck]:
        """Check numerical claims in text for consistency"""
        fact_checks = []
        
        # Extract numerical claims
        number_patterns = [
            r'\$?([\d,]+(?:\.\d{1,2})?)\s*(billion|million|thousand|B|M|K)',
            r'([\d.]+)%',
            r'revenue.*\$?([\d,]+(?:\.\d{2})?)',
            r'profit.*\$?([\d,]+(?:\.\d{2})?)',
            r'growth.*?([\d.]+)%'
        ]
        
        for pattern in number_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                claim = match.group(0)
                
                # Basic sanity checks for numerical claims
                confidence = self._validate_numerical_claim(claim)
                
                fact_checks.append(FactCheck(
                    claim=claim,
                    confidence=confidence,
                    sources=[],  # Would be populated with source verification
                    verified=confidence >= self.min_confidence,
                    evidence=[f"Numerical validation: {confidence:.2f}"],
                    timestamp=datetime.now()
                ))
        
        return fact_checks
    
    def _validate_numerical_claim(self, claim: str) -> float:
        """Validate numerical claims for reasonableness"""
        confidence = 0.7  # Base confidence
        
        # Extract number from claim
        numbers = re.findall(r'[\d,]+(?:\.\d+)?', claim)
        if not numbers:
            return 0.3
        
        try:
            # Get the main number
            main_number = float(numbers[0].replace(',', ''))
            
            # Check for unrealistic values
            if 'billion' in claim.lower():
                # Market caps above $10 trillion are suspicious
                if main_number > 10000:
                    confidence -= 0.3
            elif 'million' in claim.lower():
                # Revenue in millions above $1 million is normal
                if main_number > 1000000:
                    confidence -= 0.2
            
            # Check percentage claims
            if '%' in claim:
                if main_number > 100:
                    confidence -= 0.4  # Percentages over 100% are suspicious
                elif main_number > 1000:
                    confidence = 0.1  # Very suspicious
            
            return max(0.1, confidence)
            
        except ValueError:
            return 0.3
    
    def cross_reference_facts(self, claims: List[str], sources: List[SourceVerification]) -> Dict[str, float]:
        """Cross-reference facts across multiple sources"""
        fact_confidence = {}
        
        for claim in claims:
            # Simple keyword matching across sources
            supporting_sources = 0
            total_credibility = 0
            
            for source in sources:
                if source.is_reliable:
                    # In a real implementation, we would fetch and analyze source content
                    # For now, we'll use a simplified approach
                    supporting_sources += 1
                    total_credibility += source.credibility_score
            
            if supporting_sources > 0:
                avg_credibility = total_credibility / supporting_sources
                # More sources increase confidence
                source_factor = min(1.0, supporting_sources / 3.0)
                confidence = avg_credibility * source_factor
            else:
                confidence = 0.3
            
            fact_confidence[claim] = confidence
        
        return fact_confidence
    
    def calculate_overall_confidence(self, 
                                   sources: List[SourceVerification], 
                                   fact_checks: List[FactCheck],
                                   content_length: int) -> Dict[str, Any]:
        """Calculate overall confidence score for the analysis"""
        
        # Source reliability score
        reliable_sources = [s for s in sources if s.is_reliable]
        source_score = len(reliable_sources) / max(1, len(sources)) if sources else 0.3
        
        # Average source credibility
        if sources:
            avg_credibility = sum(s.credibility_score for s in sources) / len(sources)
        else:
            avg_credibility = 0.3
        
        # Fact check score
        if fact_checks:
            verified_facts = [f for f in fact_checks if f.verified]
            fact_score = len(verified_facts) / len(fact_checks)
        else:
            fact_score = 0.7  # No specific claims to verify
        
        # Content length factor (longer content generally more comprehensive)
        length_factor = min(1.0, content_length / 2000)  # Normalize to 2000 chars
        
        # Calculate weighted overall confidence
        overall_confidence = (
            source_score * 0.4 +
            avg_credibility * 0.3 +
            fact_score * 0.2 +
            length_factor * 0.1
        )
        
        return {
            'overall_confidence': round(overall_confidence, 2),
            'source_score': round(source_score, 2),
            'avg_credibility': round(avg_credibility, 2),
            'fact_score': round(fact_score, 2),
            'length_factor': round(length_factor, 2),
            'reliable_sources': len(reliable_sources),
            'total_sources': len(sources),
            'verified_facts': len([f for f in fact_checks if f.verified]),
            'total_fact_checks': len(fact_checks),
            'confidence_level': self._get_confidence_level(overall_confidence)
        }
    
    def _get_confidence_level(self, confidence: float) -> str:
        """Get human-readable confidence level"""
        if confidence >= 0.9:
            return "Very High"
        elif confidence >= 0.8:
            return "High" 
        elif confidence >= 0.7:
            return "Good"
        elif confidence >= 0.6:
            return "Moderate"
        elif confidence >= 0.5:
            return "Low"
        else:
            return "Very Low"
    
    def generate_reliability_report(self, 
                                  content: str,
                                  sources: List[Dict[str, Any]],
                                  analysis_type: str) -> Dict[str, Any]:
        """Generate comprehensive reliability report"""
        
        # Verify sources
        verified_sources = self.verify_sources(sources)
        
        # Check numerical claims
        fact_checks = self.check_numerical_claims(content)
        
        # Calculate overall confidence
        confidence_metrics = self.calculate_overall_confidence(
            verified_sources, fact_checks, len(content)
        )
        
        # Generate warnings
        warnings = []
        if confidence_metrics['overall_confidence'] < self.min_confidence:
            warnings.append("Overall confidence below threshold - verify independently")
        
        unreliable_sources = [s for s in verified_sources if not s.is_reliable]
        if unreliable_sources:
            warnings.append(f"{len(unreliable_sources)} sources have low credibility")
        
        unverified_facts = [f for f in fact_checks if not f.verified]
        if unverified_facts:
            warnings.append(f"{len(unverified_facts)} numerical claims need verification")
        
        return {
            'analysis_type': analysis_type,
            'timestamp': datetime.now().isoformat(),
            'confidence_metrics': confidence_metrics,
            'verified_sources': [
                {
                    'domain': s.domain,
                    'credibility_score': s.credibility_score,
                    'authority': s.domain_authority,
                    'reliable': s.is_reliable,
                    'notes': s.verification_notes
                }
                for s in verified_sources
            ],
            'fact_checks': [
                {
                    'claim': f.claim,
                    'confidence': f.confidence,
                    'verified': f.verified,
                    'evidence': f.evidence
                }
                for f in fact_checks
            ],
            'warnings': warnings,
            'recommendations': self._generate_recommendations(confidence_metrics, warnings)
        }
    
    def _generate_recommendations(self, confidence_metrics: Dict[str, Any], warnings: List[str]) -> List[str]:
        """Generate recommendations based on reliability analysis"""
        recommendations = []
        
        if confidence_metrics['overall_confidence'] < 0.7:
            recommendations.append("Seek additional sources to verify key claims")
        
        if confidence_metrics['reliable_sources'] < 3:
            recommendations.append("Consult more authoritative sources")
        
        if confidence_metrics['fact_score'] < 0.8:
            recommendations.append("Verify numerical claims with original sources")
        
        if len(warnings) > 2:
            recommendations.append("Exercise caution - multiple reliability concerns identified")
        
        if not recommendations:
            recommendations.append("Analysis meets reliability standards")
        
        return recommendations

# Global anti-hallucination guard instance
anti_hallucination_guard = AntiHallucinationGuard()

def verify_analysis_reliability(content: str, sources: List[Dict[str, Any]], analysis_type: str) -> Dict[str, Any]:
    """Verify the reliability of business analysis"""
    return anti_hallucination_guard.generate_reliability_report(content, sources, analysis_type)

def get_source_credibility_scores(sources: List[Dict[str, Any]]) -> Dict[str, float]:
    """Get credibility scores for sources"""
    verified = anti_hallucination_guard.verify_sources(sources)
    return {s.url: s.credibility_score for s in verified}

# Export main components
__all__ = [
    "AntiHallucinationGuard",
    "FactCheck",
    "SourceVerification",
    "anti_hallucination_guard",
    "verify_analysis_reliability",
    "get_source_credibility_scores"
]