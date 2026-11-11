package tensor

import "fmt"

func Fused0175ReluAddLog10(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("relu_add_log10: nil operand")
	}
	r := x.Add(y)
	r = r.ReLU()
	r = r.Log10()
	return r, nil
}
func Fused0175ReluAddLog10Name() string { return "relu_add_log10" }
