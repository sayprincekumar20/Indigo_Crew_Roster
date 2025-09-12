# app/api/debug.py
from fastapi import APIRouter, HTTPException
from datetime import date, datetime
import pandas as pd
from app.data.load_data import data_loader
import logging

logger = logging.getLogger(__name__)
router = APIRouter()

@router.get("/debug/date-issues")
async def debug_date_issues():
    """Debug endpoint to identify date comparison issues"""
    try:
        debug_info = {}
        
        # Check if data is loaded
        debug_info['data_loaded'] = data_loader.flights_df is not None
        
        if data_loader.flights_df is not None:
            # Check the data types
            debug_info['flights_df_dtypes'] = str(data_loader.flights_df.dtypes.to_dict())
            
            # Check Date column specifically
            if 'Date' in data_loader.flights_df.columns:
                debug_info['date_column_type'] = str(data_loader.flights_df['Date'].dtype)
                debug_info['date_column_sample'] = data_loader.flights_df['Date'].head(3).tolist()
                debug_info['date_column_sample_types'] = [str(type(x)) for x in data_loader.flights_df['Date'].head(3)]
            
            # Test date comparison
            test_date = date(2023, 10, 1)
            debug_info['test_date'] = str(test_date)
            debug_info['test_date_type'] = str(type(test_date))
            
            # Try the comparison that's failing
            try:
                mask = data_loader.flights_df['Date'] >= test_date
                debug_info['comparison_works'] = True
                debug_info['comparison_result_count'] = mask.sum()
            except Exception as e:
                debug_info['comparison_works'] = False
                debug_info['comparison_error'] = str(e)
                
            # Try with proper conversion
            try:
                test_date_dt = pd.to_datetime(test_date)
                mask = data_loader.flights_df['Date'] >= test_date_dt
                debug_info['converted_comparison_works'] = True
                debug_info['converted_comparison_result_count'] = mask.sum()
            except Exception as e:
                debug_info['converted_comparison_works'] = False
                debug_info['converted_comparison_error'] = str(e)
        
        return debug_info
        
    except Exception as e:
        logger.error(f"Debug error: {e}")
        raise HTTPException(status_code=500, detail=f"Debug error: {str(e)}")

# Add this to your main endpoints file
