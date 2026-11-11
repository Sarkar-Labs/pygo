package tensor

import "fmt"

func Fused1520NegAddLog2(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("neg_add_log2: nil operand")
	}
	r := x.Add(y)
	r = r.Neg()
	r = r.Log2()
	return r, nil
}
func Fused1520NegAddLog2Name() string { return "neg_add_log2" }
