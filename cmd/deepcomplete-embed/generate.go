package main

import (
	"fmt"
	"io"
	"net/http"
	"os"
	"runtime"
)

func main() {
	ollamaURL := getOllamaDownloadURL()
	if ollamaURL == "" {
		fmt.Println("Unsupported OS for automatic Ollama download. Please download manually and place in PATH.")
		return
	}

	resp, err := http.Get(ollamaURL)
	if err != nil {
		fmt.Println("Error downloading Ollama:", err)
		return
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		fmt.Println("Error downloading Ollama, status code:", resp.StatusCode)
		return
	}

	binaryData, err := io.ReadAll(resp.Body)
	if err != nil {
		fmt.Println("Error reading Ollama binary:", err)
		return
	}

	// For now, let's just write the bytes to a file for demonstration.
	// In a real scenario, you'd embed this into your Go binary or use memfd_create.
	outputPath := "ollama_binary"
	if runtime.GOOS == "windows" {
		outputPath = "ollama_binary.exe"
	}

	err = os.WriteFile(outputPath, binaryData, 0755)
	if err != nil {
		fmt.Println("Error writing Ollama binary to file:", err)
		return
	}

	fmt.Printf("Ollama binary downloaded to %s. You would embed this into your library.\n", outputPath)
	fmt.Println("For embedding, you'd convert this binary to a byte slice in Go.")
	fmt.Println("Example: `ollamaBinary = []byte{ /* ... bytes from ollama_binary ... */ }`")
}

func getOllamaDownloadURL() string {
	osName := runtime.GOOS
	arch := runtime.GOARCH

	switch osName {
	case "linux":
		if arch == "amd64" {
			return "https://ollama.ai/install/linux/ollama-linux-amd64" // Example URL, check official Ollama site
		} else if arch == "arm64" {
			return "https://ollama.ai/install/linux/ollama-linux-arm64" // Example URL, check official Ollama site
		}
	case "darwin":
		if arch == "amd64" {
			return "https://ollama.ai/install/macos/ollama-darwin-amd64" // Example URL, check official Ollama site
		} else if arch == "arm64" {
			return "https://ollama.ai/install/macos/Ollama-darwin-arm64.zip" // Example URL, check official Ollama site (might need to unzip)
		}
	case "windows":
		if arch == "amd64" {
			return "https://ollama.ai/install/windows/OllamaSetup.exe" // Example URL, check official Ollama site
		}
	}
	return "" // Unsupported OS/Arch
}
