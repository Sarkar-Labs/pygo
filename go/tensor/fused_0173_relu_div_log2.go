package tensor

import "fmt"

func Fused0173ReluDivLog2(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("relu_div_log2: nil operand")
	}
	r := x.Div(y)
	r = r.ReLU()
	r = r.Log2()
	return r, nil
}
func Fused0173ReluDivLog2Name() string { return "relu_div_log2" }
