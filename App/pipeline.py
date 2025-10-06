from Transcription_parakeet.Src.Parakeet import (
	validate_paths,
	load_model,
	transcribe_files,
	print_results,
	_result_to_dict,
)


def run_pipeline(files: list[str], model: str | None = None, batch_size: int = 1):

	paths = validate_paths(files)
	if not paths:
		raise SystemExit(2)

	model_name = model or "nvidia/parakeet-tdt-0.6b-v2"
	m = load_model(model_name)
	results = transcribe_files(m, paths, batch_size=batch_size)
	print_results(paths, results)

	# Convert to structured, serializable output
	structured = [_result_to_dict(paths[i], r) for i, r in enumerate(results)]
	return structured
