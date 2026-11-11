package tensor

import "fmt"

func Fused0621ExpSubLog2(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("exp_sub_log2: nil operand")
	}
	r := x.Sub(y)
	r = r.Exp()
	r = r.Log2()
	return r, nil
}
func Fused0621ExpSubLog2Name() string { return "exp_sub_log2" }
