package main

import (
	"fmt"

	"github.com/pygo"
)

func main() {
	a := pygo.FromSlice([]float32{579}, pygo.Shape{1})
	b := pygo.FromSlice([]float32{3468}, pygo.Shape{1})
	c := a.Add(b)
	fmt.Printf("579 + 3468 = %.0f\n", c.Item())
}
