import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.compute as pc
import os

def update_market_cache_efficiently():
    # 1. Define file paths
    main_file = '/home/talal/gamma_markets_all_tokens.parquet'
    cache_file = '/home/talal/data-cache/polymarket_cache/gamma_markets_all_tokens.parquet'
    temp_file = '/home/talal/data-cache/polymarket_cache/temp_cache.parquet'
    
    print("1. Extracting existing market IDs...")
    cache_ids_table = pq.read_table(cache_file, columns=['market_id'])
    
    # Per your instruction, format the IDs as integers for perfectly safe matching.
    # String -> Float -> Int handles variations like "123" and "123.0" safely
    cache_float = pc.cast(cache_ids_table['market_id'], pa.float64(), safe=False)
    cache_int = pc.cast(cache_float, pa.int64(), safe=False)
    
    # Drop any nulls from failed casts and build our set
    existing_ids = set(pc.drop_null(cache_int).to_pylist())
    print(f"Found {len(existing_ids)} existing markets in cache.")
    
    id_array = pa.array(list(existing_ids))
    new_markets_added = 0
    
    try:
        with pq.ParquetFile(cache_file) as cache_pf:
            schema = cache_pf.schema_arrow
            
            with pq.ParquetWriter(temp_file, schema) as writer:
                print("2. Copying existing cache to temporary file in chunks...")
                for batch in cache_pf.iter_batches(batch_size=50000):
                    writer.write_batch(batch)
                    
                print("3. Scanning main file for new markets...")
                with pq.ParquetFile(main_file) as main_pf:
                    for i, batch in enumerate(main_pf.iter_batches(batch_size=50000)):
                        
                        # Cast main file IDs to integer to ensure a 1:1 match
                        batch_ids_float = pc.cast(batch['market_id'], pa.float64(), safe=False)
                        batch_ids_int = pc.cast(batch_ids_float, pa.int64(), safe=False)
                        
                        # Filter A: Is this market_id completely new?
                        mask_new = pc.invert(pc.is_in(batch_ids_int, value_set=id_array))
                        
                        # Filter B: Volume > 0 (Reverted to the simple, clean logic)
                        numeric_vol = pc.cast(batch['volume'], pa.float64(), safe=False)
                        clean_vol = pc.fill_null(numeric_vol, 0.0)
                        mask_volume = pc.greater(clean_vol, 0.0)
                        
                        # Combine masks
                        combined_mask = pc.and_(mask_new, mask_volume)
                        
                        # --- DIAGNOSTICS ---
                        if i % 10 == 0:
                            # Count how many rows evaluate to True for each mask
                            new_ids_count = pc.sum(pc.cast(mask_new, pa.int64())).as_py()
                            valid_vol_count = pc.sum(pc.cast(mask_volume, pa.int64())).as_py()
                            print(f"  Chunk {i} Stats -> New IDs: {new_ids_count} | IDs with Vol > 0: {valid_vol_count}")
                        
                        # Filter the batch natively
                        new_markets_batch = batch.filter(combined_mask)
                        
                        # 4. Write new markets, update tracking set
                        if new_markets_batch.num_rows > 0:
                            writer.write_batch(new_markets_batch)
                            new_markets_added += new_markets_batch.num_rows
                            
                            # Extract the new integer IDs to update our tracking set
                            new_ids_float = pc.cast(new_markets_batch['market_id'], pa.float64(), safe=False)
                            new_ids_int = pc.cast(new_ids_float, pa.int64(), safe=False)
                            existing_ids.update(pc.drop_null(new_ids_int).to_pylist())
                            
                            id_array = pa.array(list(existing_ids))

    except Exception as e:
        print(f"\nError encountered: {e}")
        print("Cleaning up temporary file...")
        if os.path.exists(temp_file):
            os.remove(temp_file)
        raise

    print(f"\n4. Finished processing! Added {new_markets_added} new markets.")
    
    # 5. Swap the files safely
    if new_markets_added > 0:
        os.replace(temp_file, cache_file)
        print("Success: Cache file overwritten with the new updated data.")
    else:
        if os.path.exists(temp_file):
            os.remove(temp_file)
        print("No new markets found. Original cache kept exactly as is.")

if __name__ == "__main__":
    update_market_cache_efficiently()
