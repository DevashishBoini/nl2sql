"""
Type aliases for the NL2SQL system.

Provides reusable, descriptive type aliases for common patterns
to improve code readability and type safety.
"""

from typing import Dict, Set


# Table-to-columns mapping: {table_name: {column_name, ...}}
TableColumnsMap = Dict[str, Set[str]]

# Table scores for ranking: {table_name: score}
TableScoresMap = Dict[str, float]

# Table descriptions: {table_name: description}
TableDescriptionsMap = Dict[str, str]
