package main

import (
	"fmt"
	"io"
	"net/http"
	"os"
	"os/exec"
	"runtime"
)

func main() {
	ollamaURL := getOllamaDownloadURL()

	outputPath := "ollama_binary"
	var installedPath string
	var binaryData []byte
	var err error

	if ollamaURL == "" {
		fmt.Println("Unsupported OS for automatic Ollama download. Please download manually and place in PATH.")
		os.Exit(1)
	}

	if installedPath, err = exec.LookPath("ollama"); err == nil {
		//then we use installed path
		fmt.Println("installed path %s", installedPath)
		/// Then here we take the path and create the binary thingy
	} else {
		fmt.Println("attempting to download from %s", ollamaURL)
		// Not local? Lets download it
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

		binaryData, err = io.ReadAll(resp.Body)
		if err != nil {
			fmt.Println("Error reading Ollama binary:", err)
			return
		}
	}

	// For now, let's just write the bytes to a file for demonstration.
	// In a real scenario, you'd embed this into your Go binary or use memfd_create.
	if installedPath != "" {
		outputPath = installedPath
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
}

func getOllamaDownloadURL() string {
	osName := runtime.GOOS
	arch := runtime.GOARCH

	switch osName {
	case "linux":
		if arch == "amd64" && arch == "arm64" {
			return "https://ollama.com/download/ollama-linux-" + arch + ".tgz"
		}
	case "darwin":
		if arch == "amd64" {
			return "https://ollama.com/download/Ollama-darwin.zip"
		} else if arch == "arm64" {
			return "https://ollama.com/download/Ollama-darwin.zip"
		}
	case "windows":
		if arch == "amd64" {
			return "https://ollama.com/download/OllamaSetup.exe"
		}
	}
	return "" // Unsupported OS/Arch
}
