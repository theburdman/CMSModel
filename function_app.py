import azure.functions as func
import logging
import sys
from datetime import datetime

# Import the main processing logic
from master import run_cms_pipeline

# Import container trigger logic
from container_trigger import trigger_container

app = func.FunctionApp()

@app.schedule(
    schedule="0 0 18 1 * *",  # Monthly: 6 PM on the 1st of every month (Cron: sec min hour day month dayofweek)
    arg_name="myTimer", 
    run_on_startup=False,  # Set to True for testing to run on deployment
    use_monitor=False
)
def korteCMSFunction(myTimer: func.TimerRequest) -> None:
    """
    Azure Function that processes CMS HCRIS data monthly.
    
    Schedule: Monthly on the 1st at 6 PM UTC
    For testing, you can use: "0 */5 * * * *" (every 5 minutes)
    
    Cron format: {second} {minute} {hour} {day} {month} {day-of-week}
    """
    utc_timestamp = datetime.utcnow().replace(tzinfo=None).isoformat()
    
    if myTimer.past_due:
        logging.warning('The timer is past due!')
    
    logging.info(f'CMS Function triggered at: {utc_timestamp}')
    
    try:
        # Run the main CMS data pipeline
        logging.info('Starting CMS data pipeline execution...')
        run_cms_pipeline()
        logging.info('CMS data pipeline completed successfully.')
        
    except Exception as e:
        logging.error(f'Error in CMS pipeline: {str(e)}', exc_info=True)
        raise  # Re-raise to mark function execution as failed
    
    logging.info(f'CMS Function completed at: {datetime.utcnow().isoformat()}')


# For testing locally or on-demand execution via HTTP trigger (optional)
@app.route(route="cms_manual_trigger", auth_level=func.AuthLevel.FUNCTION)
def cms_manual_trigger(req: func.HttpRequest) -> func.HttpResponse:
    """
    HTTP endpoint for manual/on-demand execution during testing.
    Call this endpoint to run the pipeline without waiting for the timer.
    """
    logging.info('Manual trigger received via HTTP.')
    
    try:
        run_cms_pipeline()
        return func.HttpResponse(
            "CMS pipeline executed successfully.",
            status_code=200
        )
    except Exception as e:
        logging.error(f'Error in manual execution: {str(e)}', exc_info=True)
        return func.HttpResponse(
            f"Error executing pipeline: {str(e)}",
            status_code=500
        )


# ===== CONTAINER TRIGGER FUNCTIONS =====
# These functions trigger a Docker container to run the pipeline
# This avoids the Azure Functions package size limitations

@app.schedule(
    schedule="0 0 3 1 * *",  # Monthly: 3 AM CST on the 1st of every month
    arg_name="myTimer", 
    run_on_startup=False,
    use_monitor=False
)
def container_scheduler(myTimer: func.TimerRequest) -> None:
    """
    Timer-triggered function that creates a container instance to run the CMS pipeline.
    
    Schedule: Monthly on the 1st at 3 AM CST (9 AM UTC)
    
    This lightweight function just triggers the container - all the heavy ML processing
    happens in the container, avoiding Azure Functions package size limits.
    """
    utc_timestamp = datetime.utcnow().replace(tzinfo=None).isoformat()
    
    if myTimer.past_due:
        logging.warning('The timer is past due!')
    
    logging.info(f'Container scheduler triggered at: {utc_timestamp}')
    
    try:
        result = trigger_container()
        logging.info(f'Container trigger result: {result}')
    except Exception as e:
        logging.error(f'Error triggering container: {str(e)}', exc_info=True)
        raise
    
    logging.info(f'Container scheduler completed at: {datetime.utcnow().isoformat()}')


@app.route(route="trigger_container", auth_level=func.AuthLevel.FUNCTION)
def manual_container_trigger(req: func.HttpRequest) -> func.HttpResponse:
    """
    HTTP endpoint for manual container triggering.
    Use this to test the container execution without waiting for the scheduled trigger.
    """
    logging.info('Manual container trigger received via HTTP.')
    
    try:
        result = trigger_container()
        return func.HttpResponse(
            f"Container triggered successfully: {result}",
            status_code=200
        )
    except Exception as e:
        logging.error(f'Error in manual container trigger: {str(e)}', exc_info=True)
        return func.HttpResponse(
            f"Error triggering container: {str(e)}",
            status_code=500
        )
