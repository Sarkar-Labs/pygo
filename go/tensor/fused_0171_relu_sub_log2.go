package tensor

import "fmt"

func Fused0171ReluSubLog2(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("relu_sub_log2: nil operand")
	}
	r := x.Sub(y)
	r = r.ReLU()
	r = r.Log2()
	return r, nil
}
func Fused0171ReluSubLog2Name() string { return "relu_sub_log2" }
