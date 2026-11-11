package tensor

import "fmt"

func Fused0260SigmoidAddLog2(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sigmoid_add_log2: nil operand")
	}
	r := x.Add(y)
	r = r.Sigmoid()
	r = r.Log2()
	return r, nil
}
func Fused0260SigmoidAddLog2Name() string { return "sigmoid_add_log2" }
