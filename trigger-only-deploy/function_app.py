"""
Lightweight Azure Function for triggering CMS pipeline container
This function does NOT contain the heavy ML packages - it just triggers the container
"""
import azure.functions as func
import logging
from datetime import datetime
from container_trigger import trigger_container

app = func.FunctionApp()

@app.schedule(
    schedule="0 0 3 1 * *",  # Monthly: 3 AM CST on the 1st of every month (9 AM UTC)
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
