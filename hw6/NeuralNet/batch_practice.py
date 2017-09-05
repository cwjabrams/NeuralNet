from batch_trainer import main


main(target_values=[0.05,0.95], epochs=12, v_decay_rate=0.9,
	w_decay_rate=0.6, decay_frequency=4)
