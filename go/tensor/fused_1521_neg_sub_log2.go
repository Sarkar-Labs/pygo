package tensor

import "fmt"

func Fused1521NegSubLog2(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("neg_sub_log2: nil operand")
	}
	r := x.Sub(y)
	r = r.Neg()
	r = r.Log2()
	return r, nil
}
func Fused1521NegSubLog2Name() string { return "neg_sub_log2" }
