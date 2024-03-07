import * as ccfapp from "@microsoft/ccf-app";


// Exporting individual constants
export const MODEL_VOTE_TABLE_NAME = "Model_Vote";

export const MODEL_VOTE_TABLE = ccfapp.typedKv(
    MODEL_VOTE_TABLE_NAME,
  ccfapp.string,
  ccfapp.json<number>(),
);

export const MODEL_NAME_A = "model_a"

export const MODEL_NAME_B = "model_b"