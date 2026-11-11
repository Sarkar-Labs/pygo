package tensor

import "fmt"

func Fused1460NegAddSqrt(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("neg_add_sqrt: nil operand")
	}
	r := x.Add(y)
	r = r.Neg()
	r = r.Sqrt()
	return r, nil
}
func Fused1460NegAddSqrtName() string { return "neg_add_sqrt" }
