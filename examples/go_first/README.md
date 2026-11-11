# Go-first PyGo example

This is the intended PyGo user experience: application code is written in Go.

```go
import "github.com/pygo"

func main() {
    a := pygo.FromSlice([]float32{579}, pygo.Shape{1})
    b := pygo.FromSlice([]float32{3468}, pygo.Shape{1})
    c := a.Add(b)
}
```

The application does not need to call Python or C++ directly. PyGo's Go API
and bridge/runtime select the appropriate backend.
