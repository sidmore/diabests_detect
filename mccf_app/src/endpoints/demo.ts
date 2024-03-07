import * as ccfapp from "@microsoft/ccf-app";
import { MODEL_VOTE_TABLE, MODEL_NAME_A, MODEL_NAME_B } from '../constants'

interface ModelVoteRequest {
modelName: string;
}

interface ModelGetResponse {
  model: string;
  }
  

export function submitVote(request: ccfapp.Request<ModelVoteRequest>): ccfapp.Response<number> {
    // access request details
    let body;
    try {
      body = request.body.json();
    } catch {
      return {
        statusCode: 400,
      };
    }
  
    const modelName = body.modelName;

    // ModelName either model_a or model_b
    if (modelName.toLowerCase() != "model_a" || modelName.toLowerCase() != "model_b")
    {
        return {
            statusCode: 400,
        }
    }

    let newVotes = 0;
    let currentVotes = MODEL_VOTE_TABLE.get(modelName) || 0;

    // currentVotes is defined, so we can safely update votes
    newVotes = currentVotes + 1;
    MODEL_VOTE_TABLE.set(modelName, newVotes);

    return {
        body: newVotes,
        statusCode: 200
    }
}

export function getModel(request: ccfapp.Request): ccfapp.Response<ModelGetResponse> {
    // access request details
    let body;
    try {
      body = request.body.json();
    } catch {
      return {
        statusCode: 400,
      };
    }
  
    const modelName = body.modelName;

    // ModelName either model_a or model_b
    if (modelName.toLowerCase() != "model_a" || modelName.toLowerCase() != "model_b")
    {
        return {
            statusCode: 400,
        }
    }

    let votesModelA = MODEL_VOTE_TABLE.get(MODEL_NAME_A) || 0;
    let votesModelB = MODEL_VOTE_TABLE.get(MODEL_NAME_B) || 0;
    let selectedModel = MODEL_NAME_A;

    if (votesModelB > votesModelA)
    {
      selectedModel = MODEL_NAME_B;
    }

    return {
        body: {model: selectedModel},
        statusCode: 200
    }
}