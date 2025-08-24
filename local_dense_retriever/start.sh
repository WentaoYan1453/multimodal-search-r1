index_file=/mnt/workspace/yanwentao/retrieval/e5_Flat.index
corpus_file=/mnt/workspace/yanwentao/retrieval/wiki-18.jsonl
retriever_name=e5
retriever_path=/mnt/workspace/yanwentao/retrieval/e5-base-v2

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python /mnt/workspace/yanwentao/code/multimodal-search-r1/local_dense_retriever/retrieval_server.py --index_path $index_file \
                                            --corpus_path $corpus_file \
                                            --topk 3 \
                                            --retriever_name $retriever_name \
                                            --retriever_model $retriever_path \
                                            --faiss_gpu
