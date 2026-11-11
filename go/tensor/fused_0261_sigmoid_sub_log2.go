package tensor

import "fmt"

func Fused0261SigmoidSubLog2(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sigmoid_sub_log2: nil operand")
	}
	r := x.Sub(y)
	r = r.Sigmoid()
	r = r.Log2()
	return r, nil
}
func Fused0261SigmoidSubLog2Name() string { return "sigmoid_sub_log2" }
