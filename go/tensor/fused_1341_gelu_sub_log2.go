package tensor

import "fmt"

func Fused1341GeluSubLog2(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("gelu_sub_log2: nil operand")
	}
	r := x.Sub(y)
	r = r.GELU()
	r = r.Log2()
	return r, nil
}
func Fused1341GeluSubLog2Name() string { return "gelu_sub_log2" }
