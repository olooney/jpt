import boto3
import json
import sys

client = boto3.client('bedrock-runtime')
model_id = 'meta.llama2-13b-chat-v1'

def ask(question):
    'Streams tokens from an AWS Bedrock model.'

    # Note: llama3 uses a different, incompatible template. :(
    input_data = {
        'prompt': f"<s>[INST] {question} [/INST]",
    }
    
    streaming_response = client.invoke_model_with_response_stream(
        modelId=model_id,
        body=json.dumps(input_data),
    )

    stream = streaming_response.get('body')
    if stream:
        for event in stream:
            chunk = event.get('chunk')
            if chunk:
                try:
                    chunk_data = json.loads(chunk.get('bytes').decode())
                    print(chunk_data.get('generation'), end='')
                    sys.stdout.flush()
                except:
                    print(chunk)

if __name__ == '__main__':
    if len(sys.argv) > 1:
        ask(sys.argv[1])
    else:
        ask("Tell me about the capital of Peru?")
