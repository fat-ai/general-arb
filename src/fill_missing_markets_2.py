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
    existing_ids = set(cache_ids_table['market_id'].to_pylist())
    print(f"Found {len(existing_ids)} existing markets in cache.")
    
    # 2. Use try...except for safe cleanup on failure
    try:
        # Use context managers for proper file handle cleanup
        with pq.ParquetFile(cache_file) as cache_pf:
            schema = cache_pf.schema_arrow
            
            with pq.ParquetWriter(temp_file, schema) as writer:
                
                print("2. Copying existing cache to temporary file in chunks...")
                for batch in cache_pf.iter_batches(batch_size=50000):
                    writer.write_batch(batch)
                    
                print("3. Scanning main file for new markets...")
                new_markets_added = 0
                
                with pq.ParquetFile(main_file) as main_pf:
                    for i, batch in enumerate(main_pf.iter_batches(batch_size=50000)):
                        # Print progress every 10 chunks (500,000 rows)
                        if i % 10 == 0:
                            print(f"  Scanning chunk {i}...")
                        
                        # Filter A: Is this market_id completely new?
                        # Using PyArrow native compute
                        id_array = pa.array(list(existing_ids))
                        mask_new = pc.invert(pc.is_in(batch['market_id'], value_set=id_array))
                        
                        # Filter B: Handle volume safely and check > 0 for fractional volumes
                        # We handle nulls, then safely cast to float64
                        safe_volume = pc.if_else(pc.is_null(batch['volume']), 0.0, batch['volume'])
                        
                        # Note: If your volume strings contain characters like 'N/A' or letters, 
                        # pure PyArrow compute cast might throw an error. 
                        numeric_vol = pc.cast(safe_volume, pa.float64(), safe=False)
                        mask_volume = pc.greater(numeric_vol, 0.0)
                        
                        # Combine masks
                        combined_mask = pc.and_(mask_new, mask_volume)
                        
                        # Filter the batch natively
                        new_markets_batch = batch.filter(combined_mask)
                        
                        # 4. If we found new markets, write them and update our tracking set
                        if new_markets_batch.num_rows > 0:
                            writer.write_batch(new_markets_batch)
                            new_markets_added += new_markets_batch.num_rows
                            
                            # Prevent duplicates from later chunks by updating the set
                            existing_ids.update(new_markets_batch['market_id'].to_pylist())

    except Exception as e:
        print(f"\nError encountered: {e}")
        print("Cleaning up temporary file...")
        if os.path.exists(temp_file):
            os.remove(temp_file)
        # Re-raise the exception so you can debug the actual failure
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
