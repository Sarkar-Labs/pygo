package tensor

import "fmt"

func Fused0720SinAddRelu(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sin_add_relu: nil operand")
	}
	r := x.Add(y)
	r = r.Sin()
	r = r.ReLU()
	return r, nil
}
func Fused0720SinAddReluName() string { return "sin_add_relu" }
