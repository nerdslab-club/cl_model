# # tgt_output_probability is not a tensor
# tgt_output_probability = output_vocab_builder.batch_encoder(tgt_batch, is_only_probability=True)
# # Removing the <BOS> output token
# # tgt_output_probability = tgt_output_probability[:, 1:]
#
# # batch_route_ids and output_logits, tgt_output_probability
# print("output logits", len(output_logits), len(output_logits[0]), output_logits[0][0].shape)
# print("batch_route_ids", len(batch_route_ids), len(batch_route_ids[0]), batch_route_ids[0][0])
# print("tgt_output_probability", len(tgt_output_probability), len(tgt_output_probability[0]),
#       tgt_output_probability[0][0])
# # TODO calculate each output classification head batch

# # Initializing the output losses map
# output_losses_map = create_initial_output_losses_map(
#     output_classification_head_index_list=
#     category_vocab_builder.index_to_output_token_classification_head_vocab_item.keys(),
# )
# # Populating the output losses map
# # For the route_ids and logits we need to discard the last prediction as that is garbage
# # Check batch_builder line #253 for detailed explanation.
# # While for the target output probability we need to discard the first <BOS> token
# # As it will be provided initially
# for sequence_route_ids, sequence_logits, sequence_tgt_probability in zip(batch_route_ids, output_logits, tgt_output_probability):
#     for route_id, logit, tgt_probability in zip(sequence_route_ids[:-1], sequence_logits[:-1], sequence_tgt_probability[1:]):
#         current_loss_map = output_losses_map[route_id]
#         current_loss_map[IS_ITEM_PRESENT] = True
#         current_loss_map[OUTPUT_LOGITS] = logit.unsqueeze(0) if current_loss_map[OUTPUT_LOGITS] is None else torch.cat(
#                 (current_loss_map[OUTPUT_LOGITS], logit.unsqueeze(0)),
#                 dim=0,
#             )
#         current_loss_map[TARGET_OUTPUT_PROBABILITY].append(tgt_probability)
#         output_losses_map[route_id] = current_loss_map
#
# batch_output_loss = {}
# batch_output_accuracy = {}
# for index, output_loss_map in output_losses_map.items():
#     if output_loss_map[IS_ITEM_PRESENT]:
#         if index == 0 or index == 2:
#             continue
#         print(index)
#         logits = output_loss_map[OUTPUT_LOGITS].unsqueeze(0)
#         tgt_probability = torch.tensor(output_loss_map[TARGET_OUTPUT_PROBABILITY]).unsqueeze(0)
#
#         print(f"foofoo calculated logits shape {logits.shape}")
#         print(f"foofoo calculated tgt_probability shape {tgt_probability.shape}")
#         current_batch_output_loss = criterion(
#             logits.contiguous().permute(0, 2, 1),
#             tgt_probability.contiguous().long(),
#         )
#         batch_output_loss[index] = current_batch_output_loss