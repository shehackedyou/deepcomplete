package promptformat

import "fmt"

// PromptFormat is an interface for different prompt structures.
type PromptFormat interface {
	ConstructPrompt(prompt string, prefix string, suffix string) string
	GetName() string // Optional, for identification purposes
}

// GeneralTextPromptFormat is for standard text generation prompts.
type GeneralTextPromptFormat struct{}

func (f GeneralTextPromptFormat) ConstructPrompt(prompt string, prefix string, suffix string) string {
	// For general text, we can just use the prompt directly, or add some basic context if needed.
	// For now, let's just return the prompt as is.
	return prompt
}

func (f GeneralTextPromptFormat) GetName() string {
	return "GeneralText"
}

// CodeCompletionPromptFormat is for code completion prompts, using prefix and suffix.
type CodeCompletionPromptFormat struct{}

func (f CodeCompletionPromptFormat) ConstructPrompt(prompt string, prefix string, suffix string) string {
	// Construct a prompt suitable for code completion, using prefix and suffix.
	return fmt.Sprintf("Prefix:\n%s\n\nSuffix:\n%s\n\nComplete the following code:\n%s", prefix, suffix, prompt)
}

func (f CodeCompletionPromptFormat) GetName() string {
	return "CodeCompletion"
}
