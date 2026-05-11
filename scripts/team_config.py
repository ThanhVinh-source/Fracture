TEAM_PIPELINES = {
    'market-risk-quant': {
        'trade_positions_sftp':   {'archetype': 'asymmetry',    'initial_gap': 25, 'gap_growth': 1.5},
        'positions_sftp_ingest':  {'archetype': 'fast_drifting','drift_rate':  3.5},
    },
    'aml-platform': {
        'customer_risk_features': {'archetype': 'silent',       'silent_days': [0, 3]},
        'transaction_screening':  {'archetype': 'healthy'},
    },
    'payments-platform': {
        'payment_settlements_daily': {'archetype': 'healthy'},
        'payment_reconciliation':    {'archetype': 'degrading', 'drift_rate': 1.5},
    },
    'risk-technology': {
        'grid_corehours_calc':    {'archetype': 'degrading',    'drift_rate': 2.0},
        'var_batch_processing':   {'archetype': 'bimodal'},
    },
    'regulatory-reporting': {
        'regulatory_batch_dnb':   {'archetype': 'healthy'},
        'mifid_trade_reporting':  {'archetype': 'asymmetry',    'initial_gap': 15, 'gap_growth': 0.5},
    },
    'data-engineering': {
        'raw_ingestion_pipeline': {'archetype': 'healthy'},
        'feature_store_refresh':  {'archetype': 'silent',       'silent_days': [5, 6]},
        'dbt_model_runner':       {'archetype': 'fast_drifting','drift_rate':  2.5},
    },
    'ml-platform': {
        'model_training_pipeline':{'archetype': 'bimodal'},
        'inference_batch_scoring':{'archetype': 'asymmetry',    'initial_gap': 20, 'gap_growth': 2.0},
        'feature_validation_job': {'archetype': 'healthy'},
    },
    'credit-risk': {
        'pd_model_batch':         {'archetype': 'degrading',    'drift_rate': 1.8},
        'lgd_calculation_daily':  {'archetype': 'healthy'},
        'stress_test_scenario':   {'archetype': 'silent',       'silent_days': [1, 3]},
    },
    'platform-infrastructure': {
        'kafka_compaction_job':   {'archetype': 'healthy'},
        'hdfs_replication_check': {'archetype': 'degrading',    'drift_rate': 3.0},
        'delta_vacuum_runner':    {'archetype': 'silent',       'silent_days': [6]},
    },
    'treasury-ops': {
        'fx_rate_ingestion':      {'archetype': 'healthy'},
        'liquidity_reporting':    {'archetype': 'asymmetry',    'initial_gap': 30, 'gap_growth': 1.0},
        'collateral_valuation':   {'archetype': 'fast_drifting','drift_rate':  4.0},
    },
}

CUSTOM_MARKER_PIPELINES = {
    'airflow-pipelines': {
        'etl_daily_transform': {
            'archetype':    'healthy',
            'activity_map': {
                'queued': 'SCHEDULED', 'running': 'STARTED',
                'success': 'COMPLETED', 'data_published': 'DATA_AVAILABLE',
            },
        },
        'spark_batch_job': {
            'archetype': 'degrading', 'drift_rate': 2.0,
            'activity_map': {
                'APPLICATION_START': 'SCHEDULED', 'JOB_STARTED': 'STARTED',
                'JOB_COMPLETE': 'COMPLETED', 'OUTPUT_COMMITTED': 'DATA_AVAILABLE',
            },
        },
    },
}

TEAM_OWNERS = {
    'market-risk-quant':       'market-risk@bank.com',
    'aml-platform':            'aml-platform@bank.com',
    'payments-platform':       'payments@bank.com',
    'risk-technology':         'risk-tech@bank.com',
    'regulatory-reporting':    'regulatory@bank.com',
    'data-engineering':        'data-eng@bank.com',
    'ml-platform':             'ml-platform@bank.com',
    'credit-risk':             'credit-risk@bank.com',
    'platform-infrastructure': 'platform-infra@bank.com',
    'treasury-ops':            'treasury@bank.com',
    'airflow-pipelines':       'data-eng@bank.com',
}

PIPELINE_CONSUMERS = {
    'trade_positions_sftp':      'grid-scheduler',
    'positions_sftp_ingest':     'data-warehouse',
    'customer_risk_features':    'ml-platform',
    'transaction_screening':     'compliance-reporting',
    'payment_settlements_daily': 'settlement-ops',
    'payment_reconciliation':    'finance-ops',
    'grid_corehours_calc':       'reporting-platform',
    'var_batch_processing':      'risk-reporting',
    'regulatory_batch_dnb':      'regulatory-portal',
    'mifid_trade_reporting':     'mifid-portal',
    'raw_ingestion_pipeline':    'feature-store',
    'feature_store_refresh':     'ml-platform',
    'dbt_model_runner':          'analytics-bi',
    'model_training_pipeline':   'inference-serving',
    'inference_batch_scoring':   'downstream-consumers',
    'feature_validation_job':    'model-training',
    'pd_model_batch':            'credit-reporting',
    'lgd_calculation_daily':     'risk-reporting',
    'stress_test_scenario':      'regulatory-reporting',
    'kafka_compaction_job':      'data-lake',
    'hdfs_replication_check':    'data-warehouse',
    'delta_vacuum_runner':       'data-lake',
    'fx_rate_ingestion':         'market-data-consumers',
    'liquidity_reporting':       'cfo-systems',
    'collateral_valuation':      'risk-management',
    'etl_daily_transform':       'data-warehouse',
    'spark_batch_job':           'analytics-platform',
}
